from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import product
from functions import process_estimation, process_estimation_gruesa
import json
import time
import numpy as np
import pandas as pd
import pulp as pl
import streamlit as st
import unicodedata

# -*- coding: utf-8 -*-
"""
App Streamlit – Asignación de equipos (nuevo modelo MILP)
---------------------------------------------------------
- Mantiene el input Excel **idéntico**: columnas EXACTAS `CAMPO`, `FECHA`, `RECEPCIÓN` (kg).
- Se procesa la RECEPCIÓN en **toneladas** (kg/1000) y se usa como **límite de demanda** por (campo, fecha).
- Ingresos: **ganancia por ton** (configurable) ⇒ **$ por tonelada ganancia_ton**.
- Se adapta la interfaz para el **nuevo modelo** con equipos de tipo *Sadema* y *Hydro*,
  compatibilidades por campo y recurso "Cadena" desplegable a lo más en un campo por día.

Requisitos locales:
    pip install streamlit pulp pandas numpy openpyxl xlsxwriter

Ejecutar:
    streamlit run app_streamlit_nuevo_modelo_asignacion.py
"""

def render_cplex_solution_tables(sol_dict, calendar_map=None):
    dfs = parse_cplex_solution(sol_dict, calendar_map=calendar_map)

    st.subheader("📌 Resumen")
    st.dataframe(dfs["resumen"], use_container_width=True)

    st.subheader("🔢 z por fecha y campo")
    st.dataframe(dfs["z_df"], use_container_width=True)

    st.subheader("🚦 Cronograma por equipo y día")
    st.dataframe(dfs["schedule_df"], use_container_width=True)

    if not dfs["pivot_df"].empty:
        st.subheader("📊 Cronograma (pivot) — Campo por Equipo")
        st.dataframe(dfs["pivot_df"].reset_index(), use_container_width=True)

    st.subheader("⚙️ Kg por día y por equipo (con TOTAL)")
    if not dfs["contrib_df"].empty:
        st.dataframe(dfs["pivot_con_total"].reset_index().rename(columns={"index":"FECHA"}), use_container_width=True)
    else:
        st.info("No hay kilos asignados por equipo.")

    st.subheader("⛓️ Uso de Cadena (y[f,t]=1)")
    if dfs["chain_df"].empty:
        st.info("Cadena no utilizada.")
    else:
        st.dataframe(dfs["chain_df"], use_container_width=True)

    st.subheader("🧭 Movimientos (w, Origen→Destino)")
    if dfs["moves_df"].empty:
        st.info("No hay movimientos (o no se registraron arcos w con cambio de campo).")
    else:
        st.dataframe(dfs["moves_df"], use_container_width=True)

    # Descargas
    def _csv(df):
        return df.to_csv(index=False).encode("utf-8")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("⬇️ resumen.csv", _csv(dfs["resumen"]), "resumen.csv", "text/csv")
        st.download_button("⬇️ z_por_fecha_y_campo.csv", _csv(dfs["z_df"]), "z_por_fecha_y_campo.csv", "text/csv")
        st.download_button("⬇️ cronograma.csv", _csv(dfs["schedule_df"]), "cronograma_equipos.csv", "text/csv")
    with col2:
        if not dfs["pivot_df"].empty:
            flat = dfs["pivot_df"].copy()
            flat.columns = ["__".join([c for c in col if c]) if isinstance(col, tuple) else str(col) for col in flat.columns]
            st.download_button("⬇️ cronograma_pivot.csv", _csv(flat.reset_index()), "cronograma_pivot.csv", "text/csv")
        st.download_button("⬇️ kg_por_equipo.csv", _csv(dfs["contrib_df"]), "kg_por_equipo_y_dia.csv", "text/csv")
        if not dfs["pivot_con_total"].empty:
            st.download_button("⬇️ kg_total.csv", _csv(dfs["pivot_con_total"].reset_index()), "kg_por_dia_y_equipo_con_totales.csv", "text/csv")
    with col3:
        st.download_button("⬇️ chain_use.csv", _csv(dfs["chain_df"]), "chain_use.csv", "text/csv")
        st.download_button("⬇️ movimientos.csv", _csv(dfs["moves_df"]), "movimientos_w.csv", "text/csv")

    # Excel único
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
        dfs["resumen"].to_excel(writer, sheet_name="resumen", index=False)
        dfs["z_df"].to_excel(writer, sheet_name="z", index=False)
        dfs["schedule_df"].to_excel(writer, sheet_name="cronograma", index=False)
        if not dfs["pivot_df"].empty: dfs["pivot_df"].to_excel(writer, sheet_name="cronograma_pivot")
        dfs["contrib_df"].to_excel(writer, sheet_name="kg_por_equipo", index=False)
        if not dfs["pivot_con_total"].empty: dfs["pivot_con_total"].to_excel(writer, sheet_name="kg_total")
        dfs["chain_df"].to_excel(writer, sheet_name="chain_use", index=False)
        dfs["moves_df"].to_excel(writer, sheet_name="movimientos", index=False)
    st.download_button("📒 Descargar Excel (todas las hojas)", data=xlsx_buf.getvalue(),
                       file_name="salida_cplex.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def _load_calendar_map(path: str = "inputs/calendar.json"):
    """Devuelve dict {t:int -> fecha:str}. Si no existe, devuelve {}."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # soporta lista de diccionarios [{"t":1,"fecha":"YYYY-MM-DD"}, ...]
        t_to_fecha = {}
        for r in data:
            t_to_fecha[int(r.get("t"))] = str(r.get("fecha"))
        return t_to_fecha
    except Exception:
        return {}

def _tipo_equipo(nombre: str):
    s = str(nombre).upper()
    if s.startswith("S"): return "Sadema"
    if s.startswith("H"): return "Hidro"
    return None

def parse_cplex_solution(sol_dict, calendar_map=None):
    """
    sol_dict: dict cargado del JSON de CPLEX (con 'CPLEXSolution'->'variables')
    calendar_map: dict opcional {t:int -> 'YYYY-MM-DD'}
    """
    calendar_map = calendar_map or {}

    # métricas de cabecera (si existen)
    header = (sol_dict.get("CPLEXSolution") or {}).get("header", {}) if "CPLEXSolution" in sol_dict else sol_dict.get("header", {})
    objective = None
    try:
        objective = float(header.get("objectiveValue")) if header.get("objectiveValue") is not None else None
    except Exception:
        pass

    # lee array de variables
    variables = (sol_dict.get("CPLEXSolution") or {}).get("variables", []) or sol_dict.get("variables", [])
    # dicts acumuladores
    x_list, p_list, y_list, w_list = [], [], [], []

    for var in variables:
        name = str(var.get("name", ""))
        val_raw = var.get("value", None)
        if val_raw is None:
            continue
        try:
            val = float(val_raw)
        except Exception:
            continue

        # nombre con patrón {kind}_...
        # x_TEAM_FIELD_T             -> 4 partes
        # p_TEAM_FIELD_T             -> 4 partes (valor = ton)
        # y_FIELD_T                  -> 3 partes
        # w_TEAM_FROM_TO_T           -> 5 partes
        parts = name.split("_")
        if not parts: 
            continue

        kind = parts[0]
        # Filtramos los ~0
        if kind in ("x", "y", "w") and val < 0.5:
            continue
        if kind == "p" and val <= 1e-9:
            continue

        if kind in ("x", "p"):
            if len(parts) != 4:
                # si el campo tuviera "_" fallaría, pero en tus ejemplos no ocurre
                continue
            _, team, field, t_str = parts
            t = int(t_str)
            fecha = calendar_map.get(t, t)  # fecha si existe, si no deja t
            if kind == "x":
                x_list.append({"t": t, "FECHA": fecha, "Equipo": team, "CAMPO": field, "value": 1})
            else:  # p
                tons = float(val)
                p_list.append({
                    "t": t, "FECHA": fecha, "Equipo": team, "CAMPO": field,
                    "ton": tons, "kg": tons * 1000.0
                })

        elif kind == "y":
            # y_FIELD_T
            if len(parts) != 3:
                continue
            _, field, t_str = parts
            t = int(t_str)
            fecha = calendar_map.get(t, t)
            y_list.append({"t": t, "FECHA": fecha, "CAMPO": field, "value": 1})

        elif kind == "w":
            # w_TEAM_FROM_TO_T
            if len(parts) != 5:
                continue
            _, team, f_from, f_to, t_str = parts
            t = int(t_str)
            fecha = calendar_map.get(t, t)
            move_flag = 1 if f_from != f_to else 0
            w_list.append({
                "t": t, "FECHA": fecha, "Equipo": team,
                "Origen": f_from, "Destino": f_to,
                "EsMovimiento": move_flag, "value": 1
            })

    # DataFrames base
    x_df = pd.DataFrame(x_list)
    p_df = pd.DataFrame(p_list)
    y_df = pd.DataFrame(y_list)
    w_df = pd.DataFrame(w_list)

    # ========== z por fecha y campo ========== (suma de p por CAMPO,t)
    if not p_df.empty:
        z_df = (
            p_df.groupby(["FECHA", "CAMPO"], as_index=False)
                .agg(ton=("ton", "sum"), kg=("kg", "sum"))
                .sort_values(["FECHA", "CAMPO"])
        )
        z_df.rename(columns={"ton": "z (ton)", "kg": "z (kg)"}, inplace=True)
    else:
        z_df = pd.DataFrame(columns=["FECHA", "CAMPO", "z (ton)", "z (kg)"])

    # ========== cronograma por equipo ========== (de x; marcar Activo si hubo p en ese t,team)
    if not x_df.empty:
        # Tipo por prefijo del equipo
        x_df["Tipo"] = x_df["Equipo"].map(_tipo_equipo)
        # Activo si p_df tiene kg > 0 para (FECHA, Equipo)
        if not p_df.empty:
            prod_tm = p_df.groupby(["FECHA","Equipo"])["kg"].sum()
            def _active(row):
                return 1 if prod_tm.get((row["FECHA"], row["Equipo"]), 0.0) > 1e-6 else 0
            x_df["Activo"] = x_df.apply(_active, axis=1)
        else:
            x_df["Activo"] = 0
        schedule_df = x_df.rename(columns={"CAMPO":"Campo"})[["FECHA","Tipo","Equipo","Campo","Activo"]].sort_values(["FECHA","Equipo"])
    else:
        schedule_df = pd.DataFrame(columns=["FECHA","Tipo","Equipo","Campo","Activo"])

    # ========== pivot cronograma (Campo por equipo) ==========
    if not schedule_df.empty:
        pivot_df = schedule_df.pivot(index="FECHA", columns=["Tipo","Equipo"], values="Campo")
    else:
        pivot_df = pd.DataFrame()

    # ========== kg por día y por equipo (+ TOTAL) ==========
    if not p_df.empty:
        contrib_df = p_df.groupby(["FECHA","Equipo"], as_index=False)["kg"].sum().sort_values(["FECHA","Equipo"])
        pivot_equipo = (
            contrib_df.pivot_table(index="FECHA", columns="Equipo", values="kg", aggfunc="sum")
            .sort_index().fillna(0.0)
        )
        total_row = pd.DataFrame(pivot_equipo.sum(axis=0)).T
        total_row.index = ["TOTAL"]
        pivot_con_total = pd.concat([pivot_equipo, total_row], axis=0)
    else:
        contrib_df = pd.DataFrame(columns=["FECHA","Equipo","kg"])
        pivot_con_total = pd.DataFrame()

    # ========== uso de cadena ==========
    if not y_df.empty:
        chain_df = y_df[["FECHA","CAMPO"]].sort_values(["FECHA","CAMPO"]).reset_index(drop=True)
    else:
        chain_df = pd.DataFrame(columns=["FECHA","CAMPO"])

    # ========== movimientos (solo cambios Origen!=Destino) ==========
    if not w_df.empty:
        moves_df = (
            w_df[w_df["EsMovimiento"] == 1]
            .sort_values(["FECHA","Equipo","Origen","Destino"])
            .reset_index(drop=True)
        )
    else:
        moves_df = pd.DataFrame(columns=["FECHA","Equipo","Origen","Destino"])

    resumen = pd.DataFrame({
        "Objetivo CPLEX": [objective if objective is not None else np.nan],
        "Total z (ton)": [z_df["z (ton)"].sum() if "z (ton)" in z_df else 0.0],
        "Total z (kg)":  [z_df["z (kg)"].sum()  if "z (kg)"  in z_df else 0.0],
        "Equipos activos (días)": [int(schedule_df["Activo"].sum()) if "Activo" in schedule_df else 0]
    })

    return {
        "resumen": resumen,
        "z_df": z_df,
        "schedule_df": schedule_df,
        "pivot_df": pivot_df,
        "contrib_df": contrib_df,
        "pivot_con_total": pivot_con_total,
        "chain_df": chain_df,
        "moves_df": moves_df,
    }



# --- Estado de sesión para persistir resultados entre reruns ---
if "solution" not in st.session_state:
    st.session_state.solution = None        # dict con status, FO, etc.
    st.session_state.z_df = None            # DataFrame
    st.session_state.schedule_df = None     # DataFrame
    st.session_state.contrib_df = None      # DataFrame
    st.session_state.chain_df = None        # DataFrame
    st.session_state.meta = {}              # {t_to_fecha, S_list, H_list, team_type}

# ============================
# 1) NUEVO MODELO (tal como lo definiste)
# ============================
Field = str
Team  = str
Day   = int

@dataclass
class Instance:
    fields: List[Field]
    teams: List[Team]
    days: List[Day]
    capacity_tpd: Dict[Team, float]                 # ton/día por equipo
    start_field: Dict[Team, Field]                  # campo inicial por equipo
    move_cost: Dict[Tuple[Field, Field], float]     # costo de mover (origen, destino)
    revenue_per_ton: float = 9000.0                 # $/ton
    demand_ton: Optional[Dict[Tuple[Field, Day], float]] = None  # límite por campo y día (opcional)
    max_teams_per_field: Optional[Dict[Tuple[Field, Day], int]] = None  # opcional
    instalation_cost:float=40000
    # Tipos y flags de admisión por campo
    # field_allow[f] = {"Sadema": bool, "Hidro": bool, "Chain": bool}
    team_type: Dict[Team, str] = None
    field_allow: Dict[Field, Dict[str, bool]] = None
    max_moves_per_day: Optional[int] = None



def _norm_field(s: str) -> str:
    s = str(s).strip().upper()
    return ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')

def find_index_field(P: List[str], target: str) -> int:
    key = _norm_field(target)
    for i, f in enumerate(P):
        if _norm_field(f) == key:
            return i
    return 0  # fallback si no está


def default_symmetric_pairs(P: List[str], value: float = 0.0) -> pd.DataFrame:
    """Genera filas únicas (ORIGEN < DESTINO) con costo por par."""
    rows = []
    for i, a in enumerate(P):
        for j, b in enumerate(P):
            if j <= i:
                continue  # solo mitad superior
            rows.append({"ORIGEN": str(a), "DESTINO": str(b), "COSTO": float(value)})
    return pd.DataFrame(rows, columns=["ORIGEN", "DESTINO", "COSTO"])


def pairs_to_symmetric_dict(df_pairs: pd.DataFrame, P: List[str]) -> Dict[Tuple[str, str], float]:
    """Convierte filas únicas a dict simétrico, con diagonal = 0."""
    out: Dict[Tuple[str, str], float] = {}
    # diagonal en 0
    for a in P:
        out[(str(a), str(a))] = 0.0

    if df_pairs is not None and not df_pairs.empty:
        for _, r in df_pairs.iterrows():
            a = str(r["ORIGEN"])
            b = str(r["DESTINO"])
            c = float(r["COSTO"])
            out[(a, b)] = c
            out[(b, a)] = c

    # si faltó algún par off-diagonal, lo completamos con 0.0 (o el default que prefieras)
    for a in P:
        for b in P:
            if a == b:
                continue
            if (a, b) not in out:
                out[(a, b)] = out.get((b, a), 0.0)
    return out



from run_model import run

# ============================
# 2) UTILIDADES UI / TABLAS
# ============================
def _norm_field(s: str) -> str:
    s = str(s).strip().upper()
    # quitar acentos
    return ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')

HYDRO_ONLY = {
    "LAS VERTIENTES", "EL HUAICO", "EL ROSARIO", "SAN HERNAN", "LA ROBLERIA"
}

CHAIN_ONLY = {
    "EL RETORNO", "EL ESPINO"
}

BOTH_TYPES = {
    "SANTA MARGARITA","PANQUEHUE", "LOS ENCINOS", "LA ESPERANZA", "LAS CABRAS", "LIHUEIMO",
    "LA LAJUELA", "PUENTE NEGRO", "ENTRE RIOS", "RIO CLARO", "EL CONDOR",
    "EL DESCANSO", "BODEGA", "CHUMPIRRO", "GALVARINO"
}

def build_default_flags_df(P: List[str]) -> pd.DataFrame:
    """Devuelve el DataFrame de flags por campo con defaults según reglas pedidas."""
    rows = []
    for f in P:
        k = _norm_field(f)
        if k in HYDRO_ONLY:
            sadema, hidro, chain = False, True, False
        elif k in CHAIN_ONLY:
            sadema, hidro, chain = False, False, True
        elif k in BOTH_TYPES:
            sadema, hidro, chain = True, True, False
        else:
            # ✅ Por defecto: TODO en False para campos no listados
            sadema, hidro, chain = False, False, False

        # NUEVO: equipamiento fijo en campo (por defecto False)
        adema_fija = False
        hidro_fijo = False

        rows.append({
            "Campo": f,
            "Sadema": sadema,
            "Hidro": hidro,
            "Chain": chain,
            "Adema fija": adema_fija,
            "Hidromóvil fijo": hidro_fijo,
        })
    return pd.DataFrame(rows, columns=["Campo", "Sadema", "Hidro", "Chain", "Adema fija", "Hidromóvil fijo"])





def default_cost_matrix(P: List[str], value: float = 0.0) -> pd.DataFrame:
    dfm = pd.DataFrame(index=P, columns=P, dtype=float)
    for a in P:
        for b in P:
            dfm.loc[a, b] = 0.0 if a == b else float(value)
    return dfm


def matrix_to_dict(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for p in df.index:
        for q in df.columns:
            # Incluir SIEMPRE la diagonal con 0.0 para evitar KeyError al "quedarse" en el mismo campo
            if str(p) == str(q):
                out[(str(p), str(q))] = 0.0
            else:
                out[(str(p), str(q))] = float(df.loc[p, q])
    return out


# ============================
# 3) APP STREAMLIT
# ============================

st.set_page_config(page_title="MILP Asignación – Excel", layout="wide")

st.title("🧊📦 Planificador de Asignación de Hidros")
st.caption(
    "Sube un Excel con columnas EXACTAS: CAMPO, FECHA, RECEPCIÓN."
    "La RECEPCIÓN se usa como demanda (límite superior) por campo y día."
)

with st.sidebar:
    st.header("📄 Origen de datos")
    modo_datos = st.radio(
        "Elige el tipo de archivo",
        ["Recepción diaria exacta", "Estimación gruesa (con especies)"],
        index=0,
    )

    if modo_datos == "Recepción diaria exacta":
        uploaded = st.file_uploader("Sube tu archivo .xlsx", type=["xlsx"], key="uploader_exact")
        st.caption("Formato con columnas EXACTAS: CAMPO, FECHA, RECEPCIÓN.")
    else:
        uploaded = st.file_uploader(
            "Sube tu archivo .xlsx (estimación gruesa con columna ESPECIE)",
            type=["xlsx"], key="uploader_coarse"
        )
        st.caption("Debe incluir al menos: CAMPO, FECHA, RECEPCIÓN y ESPECIE.")

    st.header("📦 Capacidades y precios")
    gamma_S = st.number_input("Capacidad por Sadema (ton/día)", min_value=0.0, value=108.0)
    gamma_H = st.number_input("Capacidad por Hydromóvil (ton/día)", min_value=0.0, value=60.0)

    ganancia_ton = st.number_input("Ganancia por ton ($/ton)", min_value=0.0, value=9000.0)

    col_left, col_right = st.columns(2)
    with col_left:
        use_mov_limit = st.checkbox("Limitar movs/día", value=False)
        L_val = st.number_input(
            "Máx. movs/día",
            min_value=0, value=3, step=1,
            disabled=not use_mov_limit
        )
    with col_right:
        costo_inst = st.number_input("Costo de instalación de un equipo ($/kg)", min_value=0.0, value=40000.0)





if uploaded is None:
    if modo_datos == "Recepción diaria exacta":
        st.info("Esperando archivo Excel (.xlsx) con columnas: CAMPO, FECHA, RECEPCIÓN.")
    else:
        st.info("Esperando archivo Excel (.xlsx) con columnas: CAMPO, FECHA, RECEPCIÓN y ESPECIE.")
    st.stop()


# Leer Excel exacto
try:
    # Intento 1: hoja 'Base' (como antes)
    raw = pd.read_excel(uploaded, sheet_name='Base')
except Exception:
    # Fallback: primera hoja disponible
    raw = pd.read_excel(uploaded)

if modo_datos == "Recepción diaria exacta":
    # Validar columnas exactas
    cols_req = ["CAMPO", "FECHA", "RECEPCIÓN"]
    if not all(c in raw.columns for c in cols_req):
        st.error(f"El archivo debe contener EXACTAMENTE estas columnas: {cols_req}")
        st.stop()

    # Parseo estricto

    try:
        if 'CARTERA' in raw.columns:
            raw = raw[raw['CARTERA'] == 'PROPIOS']
        df = process_estimation(raw)
        df["CAMPO"] = df["CAMPO"].astype(str)
        df["FECHA"] = pd.to_datetime(df["FECHA"]).dt.date
        df["RECEPCIÓN"] = pd.to_numeric(df["RECEPCIÓN"], errors="raise").astype(float)
        # A ton
        df['RECEPCIÓN'] = df['RECEPCIÓN'] / 1000.0
        # Filtrar cartera si existe columna (mantener procesamiento original)
        # =======================
        # Vista previa de RECEPCIÓN (kg) por campo y fecha
        # =======================

        st.subheader("👀 Vista previa de RECEPCIÓN (kg) por campo y fecha")
        prev_df = (
            df.assign(RECEPCION_KG = df['RECEPCIÓN'] * 1000.0)
              .groupby(['FECHA', 'CAMPO'], as_index=False)['RECEPCION_KG']
              .sum()
              .sort_values(['FECHA', 'CAMPO'])
        )
        st.dataframe(prev_df, use_container_width=True)

        # También vista pivot FECHA x CAMPO
        pivot_prev = (
            prev_df.pivot(index='FECHA', columns='CAMPO', values='RECEPCION_KG')
                    .fillna(0.0)
                    .sort_index()
        )
        st.dataframe(pivot_prev, use_container_width=True)

        st.download_button(
            "Descargar vista previa (CSV)",
            data=prev_df.to_csv(index=False).encode('utf-8'),
            file_name='vista_previa_recepcion_kg.csv',
            mime='text/csv',
        )

        # Conjuntos
        P = sorted(df["CAMPO"].unique().tolist())
        fechas = sorted(df["FECHA"].unique().tolist())
        fecha_to_t = {d: i + 1 for i, d in enumerate(fechas)}
        T = list(range(1, len(fechas) + 1))

        t_to_fecha = {t: d.isoformat() for d, t in fecha_to_t.items()}

        # Demanda (ton) por (campo, día)
        demand_ton: Dict[Tuple[str, int], float] = {(p, t): 0.0 for p in P for t in T}
        for _, r in df.iterrows():
            p = str(r["CAMPO"])
            t = int(fecha_to_t[r["FECHA"]])
            ton = float(r["RECEPCIÓN"])  # ya en ton
            demand_ton[(p, t)] += ton
    except Exception as e:
        st.error(f"Error al procesar columnas: {e}")
        st.stop()


especies_sel = None
if modo_datos.startswith("Estimación"):
    cols_req = ['CAMPO', 'ESPECIE', 'FECHA','CANTIDAD']
    if "ESPECIE" not in raw.columns:
        st.error("El archivo de estimación gruesa debe incluir la columna ESPECIE.")
        st.stop()
    especies_disp = sorted(raw["ESPECIE"].astype(str).unique().tolist())
    especies_sel = st.multiselect("🧬 ESPECIES a utilizar", especies_disp, default=especies_disp)
    if not especies_sel:
        st.warning("Selecciona al menos una especie para continuar.")
        st.stop()
    raw = raw[raw["ESPECIE"].astype(str).isin(especies_sel)]
    df = process_estimation_gruesa(raw)
    st.subheader("👀 Vista previa de CANTIDAD (kg) por campo y fecha")
    prev_df = (
        df.assign(CANTIDAD_KG = df['CANTIDAD'])
          .groupby(['FECHA', 'CAMPO'], as_index=False)['CANTIDAD_KG']
          .sum()
          .sort_values(['FECHA', 'CAMPO'])
    )
    st.dataframe(prev_df, use_container_width=True)
    # También vista pivot FECHA x CAMPO
    pivot_prev = (
        prev_df.pivot(index='FECHA', columns='CAMPO', values='CANTIDAD_KG')
                .fillna(0.0)
                .sort_index()
    )
    st.dataframe(pivot_prev, use_container_width=True)
    st.download_button(
        "Descargar vista previa (CSV)",
        data=prev_df.to_csv(index=False).encode('utf-8'),
        file_name='vista_previa_recepcion_kg.csv',
        mime='text/csv',
    )
    # Conjuntos
    P = sorted(df["CAMPO"].unique().tolist())
    fechas = sorted(df["FECHA"].unique().tolist())
    fecha_to_t = {d: i + 1 for i, d in enumerate(fechas)}
    T = list(range(1, len(fechas) + 1))
    t_to_fecha = {t: d.isoformat() for d, t in fecha_to_t.items()}
    # Demanda (ton) por (campo, día)
    demand_ton: Dict[Tuple[str, int], float] = {(p, t): 0.0 for p in P for t in T}
    for _, r in df.iterrows():
        p = str(r["CAMPO"])
        t = int(fecha_to_t[r["FECHA"]])
        ton = float(r["CANTIDAD"])  # ya en ton
        demand_ton[(p, t)] += ton/1000


st.subheader("Calendario de días (t → FECHA)")
st.dataframe(pd.DataFrame({"t": T, "FECHA": [t_to_fecha[t] for t in T]}), use_container_width=True)



# =======================
# Flags de admisión por campo
# =======================
st.subheader("🌾 Flags por campo (Permisos) y 🛠️ Equipamiento fijo")
st.caption("Marca si el campo permite cada tipo de equipo y si posee una Sadema fija o un Hidromóvil fijo instalado.")

flags_df = build_default_flags_df(P)
flags_edit = st.data_editor(flags_df, use_container_width=True, num_rows="fixed")


# Costos de traslado (simétricos opcionales)
st.subheader("🚚 Costos de traslado entre campos")
use_sym_costs = st.checkbox("Usar costos SIMÉTRICOS (c[f,g] = c[g,f])", value=True)

if use_sym_costs:
    st.caption("Edita solo pares únicos (ORIGEN < DESTINO). La diagonal se fija en 0.")
    c_pairs_edit = st.data_editor(
        default_symmetric_pairs(P, 350000.0),  # << default en 350.000
        use_container_width=True,
        key="c_pairs",
    )
else:
    st.caption("Matriz completa (permite costos no simétricos). Editar solo off-diagonal.")
    c_costs_edit = st.data_editor(
        default_cost_matrix(P, 350000.0),      # << default en 350.000
        use_container_width=True,
        key="c_costs",
    )



st.subheader("🧰 Equipos e iniciales (día 1)")
nS = st.number_input("Cantidad de Sademas", min_value=1, value=1, step=1)
nH = st.number_input("Cantidad de Hydromóviles", min_value=1, value=1, step=1)
S_list = [f"S{i+1}" for i in range(int(nS))]
H_list = [f"H{i+1}" for i in range(int(nH))]

# índice por defecto = SANTA MARGARITA si existe; si no, 0
default_idx_SM = find_index_field(P, "SANTA MARGARITA")

start_sel = {}
cols = st.columns(max(1, min(4, len(S_list))))
for i, s in enumerate(S_list):
    with cols[i % len(cols)]:
        start_sel[s] = st.selectbox(f"Campo inicial de {s}", P, index=default_idx_SM, key=f"start_{s}")

cols = st.columns(max(1, min(4, len(H_list))))
for i, h in enumerate(H_list):
    with cols[i % len(cols)]:
        start_sel[h] = st.selectbox(f"Campo inicial de {h}", P, index=default_idx_SM, key=f"start_{h}")


st.subheader("🔒 Límite opcional de equipos por campo y día")
use_kmax = st.checkbox("Usar límite uniforme de equipos por campo y día", value=False)
kmax_val = st.number_input("Máximo equipos por campo y día", min_value=1, value=2, step=1) if use_kmax else None


# =======================
# 📦 Exportar inputs a JSONs (ZIP)
# =======================
import io, zipfile, json

def _costs_as_rows():
    """Normaliza costos a filas (ORIGEN, DESTINO, COSTO) desde simétrico o matriz completa."""
    rows = []
    if use_sym_costs:
        df_pairs = c_pairs_edit.copy()
        for _, r in df_pairs.iterrows():
            a = str(r["ORIGEN"]); b = str(r["DESTINO"]); c = float(r["COSTO"])
            rows.append({"origen": a, "destino": b, "costo": c})
            rows.append({"origen": b, "destino": a, "costo": c})
        # Agrega diagonal en 0
        for f in P:
            rows.append({"origen": f, "destino": f, "costo": 0.0})
    else:
        dfm = c_costs_edit.copy()
        for a in dfm.index:
            for b in dfm.columns:
                c = 0.0 if str(a) == str(b) else float(dfm.loc[a, b])
                rows.append({"origen": str(a), "destino": str(b), "costo": c})
    # Ordena por origen/destino para estabilidad
    rows.sort(key=lambda x: (x["origen"], x["destino"]))
    return rows

def _flags_as_rows():
    """Convierte flags_edit a filas JSON-friendly."""
    out = []
    for _, r in flags_edit.iterrows():
        out.append({
            "campo": str(r["Campo"]),
            "Sadema": bool(r["Sadema"]),
            "Hidro": bool(r["Hidro"]),
            "Chain": bool(r["Chain"]),
            "AdemaFija": bool(r.get("Adema fija", False)),
            "HidroFijo": bool(r.get("Hidromóvil fijo", False)),
        })
    out.sort(key=lambda x: x["campo"])
    return out

def _teams_as_rows():
    """Arma equipos con tipo, capacidad y campo inicial del día 1."""
    rows = []
    # capacidades ya vienen de inputs gamma_S / gamma_H
    for s in S_list:
        rows.append({
            "equipo": s, "tipo": "Sadema",
            "capacidad_tpd": float(gamma_S),
            "campo_inicial": str(start_sel[s]),
        })
    for h in H_list:
        rows.append({
            "equipo": h, "tipo": "Hidro",
            "capacidad_tpd": float(gamma_H),
            "campo_inicial": str(start_sel[h]),
        })
    rows.sort(key=lambda x: x["equipo"])
    return rows

def _calendar_rows():
    return [{"t": int(t), "fecha": str(t_to_fecha[t])} for t in T]

def _demand_rows():
    rows = []
    for (campo, t) in sorted(demand_ton.keys(), key=lambda x: (x[0], x[1])):
        rows.append({
            "campo": str(campo),
            "t": int(t),
            "fecha": str(t_to_fecha[int(t)]),
            "demanda_ton": float(demand_ton[(campo, t)]),
        })
    return rows

def _kmax_rows():
    if not (use_kmax and kmax_val is not None):
        return None
    return [{"campo": f, "t": int(t), "max_equipos": int(kmax_val)} for f in P for t in T]

def _params_dict():
    return {
        "ganancia_ton": float(ganancia_ton),
        "instalation_cost": float(costo_inst),
        "use_sym_costs": bool(use_sym_costs),
        "use_mov_limit": bool(use_mov_limit),
        "max_moves_per_day": int(L_val) if use_mov_limit else None,
        "gamma_S_default": float(gamma_S),
        "gamma_H_default": float(gamma_H),
    }
# =======================
# 📦 Exportar inputs → botón para guardar en 'inputs' + ZIP
# =======================
import io, zipfile, json
from pathlib import Path

# 1) Construir los payloads (desde el estado actual de la UI)
payloads = {
    "fields_flags.json": json.dumps(_flags_as_rows(), ensure_ascii=False, indent=2),
    "costs.json":        json.dumps(_costs_as_rows(), ensure_ascii=False, indent=2),
    "teams.json":        json.dumps(_teams_as_rows(), ensure_ascii=False, indent=2),
    "calendar.json":     json.dumps(_calendar_rows(), ensure_ascii=False, indent=2),
    "demand.json":       json.dumps(_demand_rows(), ensure_ascii=False, indent=2),
    "params.json":       json.dumps(_params_dict(), ensure_ascii=False, indent=2),
}
kmax_payload = _kmax_rows()
if kmax_payload is not None:
    payloads["max_teams_per_field.json"] = json.dumps(kmax_payload, ensure_ascii=False, indent=2)

st.subheader("📦 Exportar inputs")
st.caption("Guarda en ./inputs (sobrescribe si existen) y ofrece un ZIP con los mismos archivos.")

# 2) Botón: guardar en carpeta ./inputs
save_btn = st.button("🗂️ Guardar en carpeta inputs", type="primary", help="Sobrescribe si ya existen")
if save_btn:
    out_dir = Path("inputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, text in payloads.items():
        (out_dir / fname).write_text(text, encoding="utf-8")
    st.success(f"Archivos guardados en: {out_dir.resolve()}")
    st.write("Generados:", ", ".join(payloads.keys()))
    run()

# 3) Construir ZIP para descarga (siempre disponible)
zip_bytes = io.BytesIO()
with zipfile.ZipFile(zip_bytes, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
    for fname, text in payloads.items():
        zf.writestr(fname, text)

st.download_button(
    "💾 Descargar inputs (.zip)",
    data=zip_bytes.getvalue(),
    file_name="inputs_asignacion.zip",
    mime="application/zip",
    type="secondary",
)



# =======================
# Resolver
# =======================
# =======================
# Resolver (genera inputs -> run() -> lee solucion.json -> render)
# =======================
st.divider()
st.header("🧠 Resolver modelo (nuevo)")
verbose = st.checkbox("Ver log del solver", value=True)
solve_btn = st.button("🚀 Resolver MILP ahora", type="primary")

if solve_btn:
    try:
        # 1) (Re)generar inputs en ./inputs desde los payloads actuales de la UI
        out_dir = Path("inputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        for fname, text in payloads.items():
            (out_dir / fname).write_text(text, encoding="utf-8")
        st.success(f"Inputs escritos en {out_dir.resolve()}")

        # 2) Ejecutar CPLEX vía run() (del módulo run_model)
        with st.spinner("Ejecutando CPLEX con run_model.run()..."):
            try:
                run(verbose=verbose)  # si tu run soporta verbose
            except TypeError:
                run()

        # 3) Localizar y cargar solucion.json
        candidatos = [
            Path("solucion.json"),
            Path("outputs/solucion.json"),
            Path("/mnt/data/solucion.json"),
        ]
        sol_path = next((p for p in candidatos if p.exists()), None)
        if sol_path is None:
            st.error(
                "No encontré 'solucion.json'. "
                "Asegúrate de que run() lo deje en el directorio de trabajo "
                "o ajusta las rutas en el código."
            )
            st.stop()

        with open(sol_path, "r", encoding="utf-8") as f:
            sol = json.load(f)

        # 4) Mapear t->fecha desde inputs/calendar.json (si existe) y parsear
        t_to_fecha_map = _load_calendar_map("inputs/calendar.json")
        dfs = parse_cplex_solution(sol, calendar_map=t_to_fecha_map)

        # 5) Preparar bytes para descargas persistentes
        def _csv_bytes(df):
            return df.to_csv(index=False).encode("utf-8")

        xlsx_buf = io.BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
            dfs["resumen"].to_excel(writer, sheet_name="resumen", index=False)
            dfs["z_df"].to_excel(writer, sheet_name="z", index=False)
            dfs["schedule_df"].to_excel(writer, sheet_name="cronograma", index=False)
            if not dfs["pivot_df"].empty:
                dfs["pivot_df"].to_excel(writer, sheet_name="cronograma_pivot")
            dfs["contrib_df"].to_excel(writer, sheet_name="kg_por_equipo", index=False)
            if not dfs["pivot_con_total"].empty:
                dfs["pivot_con_total"].to_excel(writer, sheet_name="kg_total")
            dfs["chain_df"].to_excel(writer, sheet_name="chain_use", index=False)
            dfs["moves_df"].to_excel(writer, sheet_name="movimientos", index=False)

        downloads = {
            "resumen_csv": _csv_bytes(dfs["resumen"]),
            "z_csv": _csv_bytes(dfs["z_df"]),
            "cronograma_csv": _csv_bytes(dfs["schedule_df"]),
            "cronograma_pivot_csv": (
                dfs["pivot_df"].copy().reset_index().to_csv(index=False).encode("utf-8")
                if not dfs["pivot_df"].empty else None
            ),
            "kg_por_equipo_csv": _csv_bytes(dfs["contrib_df"]),
            "kg_total_csv": (
                dfs["pivot_con_total"].reset_index().to_csv(index=False).encode("utf-8")
                if not dfs["pivot_con_total"].empty else None
            ),
            "chain_use_csv": _csv_bytes(dfs["chain_df"]),
            "moves_csv": _csv_bytes(dfs["moves_df"]),
            "excel_xlsx": xlsx_buf.getvalue(),
        }

        # 6) Persistir y forzar rerun (render solo en la sección persistente)
        st.session_state.cplex_raw = sol
        st.session_state.cplex_dfs = dfs
        st.session_state.cplex_downloads = downloads

        st.rerun()

    except Exception as e:
        st.error(f"Ocurrió un error resolviendo con CPLEX: {e}")
        st.exception(e)



# =======================
# 🔁 Render persistente
# =======================
# ============================
# 📥 Cargar y PARSEAR solución CPLEX (x, p, y, w) → Tablas
# ============================
import json
import io
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path





# ============================
# UI: cargar solucion.json de CPLEX
# ============================
# --- Mostrar resultados persistentes si existen ---
if "cplex_dfs" in st.session_state and st.session_state.cplex_dfs is not None:
    st.info("Mostrando la última solución cargada/ejecutada (persistente).")
    dfs = st.session_state.cplex_dfs
    render_cplex_solution_tables(st.session_state.cplex_raw, calendar_map=_load_calendar_map("inputs/calendar.json"))

    # Descargas persistentes con keys FIJAS (no cambian entre reruns)
    dl = st.session_state.cplex_downloads
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ resumen.csv", dl["resumen_csv"], "resumen.csv", "text/csv", key="persist_resumen")
        st.download_button("⬇️ z_por_fecha_y_campo.csv", dl["z_csv"], "z_por_fecha_y_campo.csv", "text/csv", key="persist_z")
        st.download_button("⬇️ cronograma.csv", dl["cronograma_csv"], "cronograma_equipos.csv", "text/csv", key="persist_cron")
    with c2:
        if dl["cronograma_pivot_csv"] is not None:
            st.download_button("⬇️ cronograma_pivot.csv", dl["cronograma_pivot_csv"], "cronograma_pivot.csv", "text/csv", key="persist_pivot")
        st.download_button("⬇️ kg_por_equipo.csv", dl["kg_por_equipo_csv"], "kg_por_equipo_y_dia.csv", "text/csv", key="persist_kg_e")
        if dl["kg_total_csv"] is not None:
            st.download_button("⬇️ kg_total.csv", dl["kg_total_csv"], "kg_por_dia_y_equipo_con_totales.csv", "text/csv", key="persist_kg_t")
    with c3:
        st.download_button("⬇️ chain_use.csv", dl["chain_use_csv"], "chain_use.csv", "text/csv", key="persist_chain")
        st.download_button("⬇️ movimientos.csv", dl["moves_csv"], "movimientos_w.csv", "text/csv", key="persist_moves")
        st.download_button("📒 Excel (todas las hojas)", dl["excel_xlsx"], "salida_cplex.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="persist_excel")

st.divider()
st.header("📤 Cargar solución CPLEX (x/p/y/w)")

# intenta mapear t→fecha desde inputs/calendar.json si existe
t_to_fecha = _load_calendar_map("inputs/calendar.json")

tab1, tab2 = st.tabs(["Subir solucion.json", "Leer desde ruta local"])

with tab1:
    up = st.file_uploader("Sube tu solucion.json exportado por CPLEX", type=["json"], key="upl_cplex")
    if up is not None:
        try:
            sol = json.load(up)
            render_cplex_solution_tables(sol, calendar_map=t_to_fecha)
        except Exception as e:
            st.error(f"No se pudo parsear el JSON: {e}")

with tab2:
    ruta = st.text_input("Ruta local al JSON", value="/mnt/data/solucion.json")
    if st.button("Cargar solución desde ruta"):
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                sol = json.load(f)
            render_cplex_solution_tables(sol, calendar_map=t_to_fecha)
        except Exception as e:
            st.error(f"No se pudo leer el archivo '{ruta}': {e}")

if st.session_state.solution is not None:
    res          = st.session_state.solution
    z_df         = st.session_state.z_df
    schedule_df  = st.session_state.schedule_df
    contrib_df   = st.session_state.contrib_df
    chain_df     = st.session_state.chain_df
    meta         = st.session_state.meta
    t_to_fecha   = meta.get("t_to_fecha", {})
    S_list       = meta.get("S_list", [])
    H_list       = meta.get("H_list", [])
    team_type    = meta.get("team_type", {})

    # Cabecera / métricas
    st.success(f"Estado: {res['status']} | Valor objetivo: {res.get('objective',0):,.2f}")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Ingreso ($)", f"{(res.get('revenue') or 0):,.0f}")
    with col2: st.metric("Costo inicial ($)", f"{(res.get('cost_init') or 0):,.0f}")
    with col3: st.metric("Costo movimientos ($)", f"{(res.get('cost_move') or 0):,.0f}")

    # z por fecha/campo
    st.subheader("Cantidad procesada z[p,t] e ingreso estimado")
    st.dataframe(z_df, use_container_width=True)

    # Cronograma por equipo (tabla y pivot con estilo)
    st.subheader("🚦 Cronograma de equipos por día y campo")
    st.dataframe(schedule_df, use_container_width=True)

    pivot_df = schedule_df.pivot(index="FECHA", columns=["Tipo","Equipo"], values="Campo")
    act_pivot = (
        schedule_df.pivot(index="FECHA", columns=["Tipo","Equipo"], values="Activo")
        .reindex_like(pivot_df).fillna(0).astype(int)
    )
    def _style_active(df):
        mask = act_pivot.eq(1)
        return np.where(mask, 'background-color: #22c55e; color: white; font-weight: 600;', '')
    st.subheader("📊 Cronograma (por equipo) — verde = en uso")
    st.dataframe(pivot_df.style.apply(_style_active, axis=None), use_container_width=True)

    # Kg por equipo
    st.subheader("📋 Kg procesados por día y por equipo (con fila TOTAL)")
    if contrib_df is not None and not contrib_df.empty:
        pivot_equipo = (
            contrib_df.pivot_table(index="FECHA", columns="Equipo", values="kg", aggfunc="sum")
            .reindex(columns=(S_list + H_list), fill_value=0.0).sort_index()
        )
        total_row = pd.DataFrame(pivot_equipo.sum(axis=0)).T
        total_row.index = ["TOTAL"]
        pivot_con_total = pd.concat([pivot_equipo, total_row], axis=0)
        st.dataframe(pivot_con_total, use_container_width=True)
        # ⬇️ añade el botón de descarga persistente
        st.download_button(
            "Descargar tabla (CSV)",
            data=pivot_con_total.to_csv().encode("utf-8"),
            file_name="kg_por_dia_y_equipo_con_totales.csv",
            mime="text/csv",
            key="dl_kg_total",   # clave estable para que no se pierda en los reruns
        )

    # Uso de cadena
    st.subheader("⛓️ Uso de la Cadena (y[f,t]=1)")
    if chain_df is None or chain_df.empty:
        st.info("Cadena no utilizada.")
    else:
        st.dataframe(chain_df.sort_values(["FECHA","CAMPO"]).reset_index(drop=True), use_container_width=True)

    # Descargas (usan objetos persistidos)
    st.subheader("📥 Descargar resultados (persistentes)")
    st.download_button("Descargar z (CSV)", data=z_df.to_csv(index=False).encode("utf-8"),
                       file_name="z_por_fecha_y_campo.csv", mime="text/csv", key="dl_z")
    st.download_button("Descargar cronograma (CSV)", data=schedule_df.to_csv(index=False).encode("utf-8"),
                       file_name="cronograma_equipos.csv", mime="text/csv", key="dl_cron")
    st.download_button("Descargar kg por equipo (CSV)", data=contrib_df.to_csv(index=False).encode("utf-8"),
                       file_name="kg_por_equipo_y_dia.csv", mime="text/csv", key="dl_kg")



st.divider()
st.caption(
    "Archivo requerido con columnas EXACTAS: CAMPO, FECHA, RECEPCIÓN. "
    "Ganancia por tonelada configurable (por defecto 9). "
    "Se adapta a compatibilidades por campo y recurso Cadena (nuevo modelo)."
)
