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


def build_and_solve(inst: Instance, write_lp: bool = False):
    F, M, T = inst.fields, inst.teams, inst.days
    cap, f0, c, c_inst = inst.capacity_tpd, inst.start_field, inst.move_cost, inst.instalation_cost
    r = inst.revenue_per_ton
    L = getattr(inst, "max_moves_per_day", None)  # escalar

    # Validaciones
    for m in M:
        assert m in cap, f"Falta capacidad para equipo {m}"
        assert m in f0, f"Falta campo inicial para equipo {m}"
        assert f0[m] in F, f"Campo inicial {f0[m]} de {m} no está en la lista de campos"
        assert inst.team_type is not None and m in inst.team_type, f"Falta tipo (Sadema/Hidro) para equipo {m}"
        assert inst.team_type[m] in ("Sadema", "Hidro"), f"Tipo inválido para {m}: {inst.team_type[m]}"
    for f in F:
        assert inst.field_allow is not None and f in inst.field_allow, f"Faltan flags de admisión para campo {f}"
        for key in ("Sadema", "Hidro", "Chain"):
            assert key in inst.field_allow[f], f"Falta flag '{key}' en field_allow[{f}]"

    S = [m for m in M if inst.team_type[m] == "Sadema"]
    H = [m for m in M if inst.team_type[m] == "Hidro"]

    # --- Horizonte y función δ (tiempo de viaje): 1 si te quedas, 2 si cambias de campo
    def travel_time(f, f2):
        return 1 if f == f2 else 2
    def X(m,f,t): return x.get((m,f,t), 0)  # helper
    idx = {t: i for i, t in enumerate(T)}
    nT = len(T)

    prob = pl.LpProblem("Asignacion_Sadema_Hidromovil_Cadena_delta", pl.LpMaximize)

    # Variables
    # Construir listas de campos permitidos por tipo
    F_H = [f for f in F if inst.field_allow[f]["Hidro"]]
    F_S_plain = [f for f in F if inst.field_allow[f]["Sadema"]]
    F_S_chain = [f for f in F if inst.field_allow[f]["Chain"] and not inst.field_allow[f]["Sadema"]]
    valid_p = {p for p in P if inst}
    x = {}
    # Hidro solo donde está permitido
    for m in H:
        for f, t in product(F_H, T):
            x[m, f, t] = pl.LpVariable(f"x_{m}_{f}_{t}", 0, 1, pl.LpBinary)

    # Sadema: donde hay Sadema o (solo con cadena) donde Chain==True
    for m in S:
        for f, t in product(F_S_plain + F_S_chain, T):
            x[m, f, t] = pl.LpVariable(f"x_{m}_{f}_{t}", 0, 1, pl.LpBinary)



    #x = pl.LpVariable.dicts("x", (M, F, T), lowBound=0, upBound=1, cat=pl.LpBinary)     # 1 si m opera en f el día t
    p = pl.LpVariable.dicts("p", (M, F, T), lowBound=0, cat=pl.LpContinuous)            # ton procesadas m,f,t
    y = pl.LpVariable.dicts("y", (F, T), lowBound=0, upBound=1, cat=pl.LpBinary)        # Cadena en f,t

    # Variables de movimiento con retardos: solo si t+δ dentro del horizonte
    w_vars = {}
    for m, f, f2, t in product(M, F, F, T):
        delta = travel_time(f, f2)
        if idx[t] + delta < nT:
            w_vars[(m, f, f2, t)] = pl.LpVariable(f"w_{m}_{f}_{f2}_{t}", lowBound=0, upBound=1, cat=pl.LpBinary)

    # Objetivo
    revenue = r * pl.lpSum(p[m][f][t] for m, f, t in product(M, F, T))
    cost_init = pl.lpSum(c.get((f0[m], f), 0.0) * X(m,f,T[0]) for m in M for f in F)
    cost_move = pl.lpSum(c.get((f, f2), 0.0) * w_vars[(m, f, f2, t)] for (m, f, f2, t) in w_vars)
    cost_inst_init = c_inst * pl.lpSum(X(m,f,T[0]) for m in M for f in F if f != f0[m])
    cost_inst_move = c_inst * pl.lpSum(w_vars[(m, f, f2, t)] for (m, f, f2, t) in w_vars if f != f2)
    prob += revenue - cost_init - cost_move - cost_inst_init - cost_inst_move


    # (1) Asignación diaria
    for m, t in product(M, T):
        prob += pl.lpSum(X(m,f,t) for f in F) <= 1, f"asign_atmost1_{m}_{t}"

    # (2) Capacidad
    for m, f, t in product(M, F, T):
        prob += p[m][f][t] <= cap[m] * X(m,f,t), f"cap_{m}_{f}_{t}"

    # (3) Demanda por campo y día (opcional)
    if inst.demand_ton is not None:
        for f, t in product(F, T):
            if (f, t) in inst.demand_ton:
                prob += pl.lpSum(p[m][f][t] for m in M) <= inst.demand_ton[(f, t)], f"dem_{f}_{t}"

    # (3b) Máximo de equipos por campo y día (opcional)
    if inst.max_teams_per_field is not None:
        for f, t in product(F, T):
            if (f, t) in inst.max_teams_per_field:
                prob += pl.lpSum(X(m,f,t) for m in M) <= inst.max_teams_per_field[(f, t)], f"maxK_{f}_{t}"

    # (4) Compatibilidades
    for m, f, t in product(H, F, T):
        if not inst.field_allow[f]["Hidro"]:
            prob += X(m,f,t) <= 0, f"compat_H_{m}_{f}_{t}"
    for m, f, t in product(S, F, T):
        allow_S = inst.field_allow[f]["Sadema"]
        allow_chain = inst.field_allow[f]["Chain"]
        if not allow_S:
            if allow_chain:
                prob += X(m,f,t) <= y[f][t], f"compat_S_chain_{m}_{f}_{t}"
            else:
                prob += X(m,f,t) <= 0, f"compat_S_forbidden_{m}_{f}_{t}"

    # (5) Recurso Cadena
    for t in T:
        prob += pl.lpSum(y[f][t] for f in F) <= 1, f"cadena_cap_{t}"
    for f, t in product(F, T):
        if not inst.field_allow[f]["Chain"] or inst.field_allow[f]["Sadema"]:
            prob += y[f][t] <= 0, f"cadena_only_where_allowed_{f}_{t}"
    for f, t in product(F, T):
        if inst.field_allow[f]["Chain"] and not inst.field_allow[f]["Sadema"]:
            prob += pl.lpSum(X(m,f,t) for m in S) <= y[f][t], f"one_sadema_with_chain_{f}_{t}"

    # (6) Flujo con retardos δ
    # (6a) Salida
    for m, f, t in product(M, F, T):
        arcs_out = [w_vars[(m, f, f2, t)] for f2 in F if (m, f, f2, t) in w_vars]
        if arcs_out:
            prob += pl.lpSum(arcs_out) == X(m,f,t), f"salida_{m}_{f}_{t}"
    # (6b) Entrada
    t0 = T[0]
    for m, f2, t2 in product(M, F, T):
        if t2 == t0:
            continue
        preds = []
        for f in F:
            delta = travel_time(f, f2)
            i_prev = idx[t2] - delta
            if 0 <= i_prev < nT:
                t_prev = T[i_prev]
                if (m, f, f2, t_prev) in w_vars:
                    preds.append(w_vars[(m, f, f2, t_prev)])
        prob += pl.lpSum(preds) == X(m,f2,t2), f"entrada_{m}_{f2}_{t2}"

    # (6c) Máximo de movimientos diarios global (opcional)
    if L is not None:
        for t in T:
            arcs_t = [var for (m, f, f2, tt), var in w_vars.items() if tt == t and f2 != f]
            if arcs_t:
                prob += pl.lpSum(arcs_t) <= L, f"max_moves_day_{t}"

    if write_lp:
        prob.writeLP("modelo_asignacion_cadena_con_deltas.lp")

    # ==========================
    # Solver CBC con parámetros FIJOS
    # ==========================
    CBC_TIME_LIMIT = 300      # segundos (fijo)
    CBC_FRAC_GAP   = 0.05    # 0.5% (fijo)
    CBC_THREADS    = 0        # 0 = auto


    solver = pl.PULP_CBC_CMD(
        msg=True,          # pon False si no quieres log
        presolve=True,
        cuts=True,
        fracGap=0.1

    )
    status = prob.solve(solver)

    # ==========================
    # Recuperación de resultados
    # ==========================
    res = {
        "status": pl.LpStatus[prob.status],
        "objective": pl.value(prob.objective),
        "revenue": pl.value(revenue),
        "cost_init": pl.value(cost_init),
        "cost_move": pl.value(cost_move),
        "assignments": [],
        "production": [],
        "chain_use": [],
    }

    if pl.LpStatus[prob.status] not in ("Optimal", "Feasible"):
        return res

    for t in T:
        for m in M:
            for f in F:
                if pl.value(X(m,f,t)) > 0.5:
                    res["assignments"].append((t, m, f))
                prod_val = pl.value(p[m][f][t])
                if prod_val and prod_val > 1e-6:
                    res["production"].append((t, f, m, prod_val))
        for f in F:
            if pl.value(y[f][t]) > 0.5:
                res["chain_use"].append((t, f))

    return res



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
        rows.append({"Campo": f, "Sadema": sadema, "Hidro": hidro, "Chain": chain})
    return pd.DataFrame(rows, columns=["Campo", "Sadema", "Hidro", "Chain"])




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
    print(demand_ton)


st.subheader("Calendario de días (t → FECHA)")
st.dataframe(pd.DataFrame({"t": T, "FECHA": [t_to_fecha[t] for t in T]}), use_container_width=True)



# =======================
# Flags de admisión por campo
# =======================
st.subheader("🌾 Flags por campo (Permisos)")
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
# Resolver
# =======================
st.divider()
st.header("🧠 Resolver modelo (nuevo)")
verbose = st.checkbox("Ver log del solver", value=True)
solve_btn = st.button("🚀 Resolver MILP ahora", type="primary")

if solve_btn:
    try:
        # Mapas del modelo
        teams: List[str] = S_list + H_list
        team_type: Dict[str, str] = {**{s: "Sadema" for s in S_list}, **{h: "Hidro" for h in H_list}}
        capacity_tpd: Dict[str, float] = {**{s: float(gamma_S) for s in S_list}, **{h: float(gamma_H) for h in H_list}}
        start_field: Dict[str, str] = {m: start_sel[m] for m in teams}
        # 🔴 AQUÍ: usa pares simétricos o la matriz completa según checkbox
        if use_sym_costs:
            move_cost: Dict[Tuple[str, str], float] = pairs_to_symmetric_dict(c_pairs_edit, P)
        else:
            move_cost: Dict[Tuple[str, str], float] = matrix_to_dict(c_costs_edit)
        r_per_ton: float =  float(ganancia_ton)

        field_allow: Dict[str, Dict[str, bool]] = {}
        for _, row in flags_edit.iterrows():
            field_allow[str(row["Campo"])] = {
                "Sadema": bool(row["Sadema"]),
                "Hidro": bool(row["Hidro"]),
                "Chain": bool(row["Chain"]),
            }

        max_teams_per_field = None
        if use_kmax and kmax_val is not None:
            max_teams_per_field = {(f, t): int(kmax_val) for f in P for t in T}

        inst = Instance(
            fields=P,
            teams=teams,
            days=T,
            capacity_tpd=capacity_tpd,
            start_field=start_field,
            move_cost=move_cost,
            instalation_cost=costo_inst,
            revenue_per_ton=r_per_ton,
            demand_ton=demand_ton,
            max_teams_per_field=max_teams_per_field,
            team_type=team_type,
            field_allow=field_allow,
        )
        
        # Límite global de movimientos (opcional)
        if use_mov_limit:
            inst.max_moves_per_day = int(L_val)  # aplica a todos los días


        with st.spinner("Construyendo y resolviendo MILP..."):
            t0 = time.time()
            res = build_and_solve(inst, write_lp=False)
            t1 = time.time()

        status = res["status"]
        obj    = res["objective"] if res["objective"] is not None else 0.0
        st.success(f"Estado: {status} | Valor objetivo: {obj:,.2f} | Tiempo: {t1 - t0:.2f}s")

        # Desglose FO
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ingreso ($)", f"{(res['revenue'] or 0):,.0f}")
        with col2:
            st.metric("Costo inicial ($)", f"{(res['cost_init'] or 0):,.0f}")
        with col3:
            st.metric("Costo movimientos ($)", f"{(res['cost_move'] or 0):,.0f}")

        # =======================
        # z (kg) por campo y día + ingreso estimado
        # =======================
        prod_by_ft_ton: Dict[Tuple[int, str], float] = {}
        for (t, f, m, q) in res.get("production", []):
            prod_by_ft_ton[(t, f)] = prod_by_ft_ton.get((t, f), 0.0) + float(q)

        z_rows = []
        for t in T:
            for f in P:
                z_ton = prod_by_ft_ton.get((t, f), 0.0)
                z_kg  = z_ton 
                z_rows.append({
                    "FECHA": t_to_fecha[t],
                    "CAMPO": f,
                    "z (kg)": z_kg,
                    "Ingreso (estimado)": float(ganancia_ton) * z_kg,
                })
        z_df = pd.DataFrame(z_rows)

        st.subheader("Cantidad procesada z[p,t] y ingreso estimado")
        st.dataframe(z_df, use_container_width=True)

        # =======================
        # Cronograma de equipos por día
        # =======================
        # Mapa asignaciones (t,m)->f
        assign_map: Dict[Tuple[int, str], str] = {}
        for (t, m, f) in res.get("assignments", []):
            assign_map[(int(t), str(m))] = str(f)

        # Mapa producción por (t,m)
        prod_tm_ton: Dict[Tuple[int, str], float] = {}
        for (t, f, m, q) in res.get("production", []):
            prod_tm_ton[(int(t), str(m))] = prod_tm_ton.get((int(t), str(m)), 0.0) + float(q)

        # Filas cronograma
        rows_sched = []
        for t in T:
            fecha = t_to_fecha[t]
            for m in (S_list + H_list):
                campo = assign_map.get((t, m), None)
                activo = 1 if prod_tm_ton.get((t, m), 0.0) > 1e-6 else 0
                rows_sched.append({
                    "FECHA": fecha,
                    "Tipo": team_type[m],
                    "Equipo": m,
                    "Campo": campo,
                    "Activo": activo,
                })
        schedule_df = pd.DataFrame(rows_sched)

        st.subheader("🚦 Cronograma de equipos por día y campo")
        st.dataframe(schedule_df, use_container_width=True)

        # Vista compacta tipo calendario (pivot) con estilo verde cuando Activo==1
        pivot_df = schedule_df.pivot(index="FECHA", columns=["Tipo", "Equipo"], values="Campo")
        act_pivot = (
            schedule_df
            .pivot(index="FECHA", columns=["Tipo", "Equipo"], values="Activo")
            .reindex_like(pivot_df)
            .fillna(0)
            .astype(int)
        )

        def _style_active(df):
            mask = act_pivot.eq(1)
            return np.where(mask, 'background-color: #22c55e; color: white; font-weight: 600;', '')

        styled_pivot = pivot_df.style.apply(_style_active, axis=None)

        st.subheader("📊 Cronograma (por equipo) — verde = en uso")
        st.dataframe(styled_pivot, use_container_width=True)

        # =======================
        # Kg por día y por equipo (usando producción real por equipo)
        # =======================
        st.subheader("📋 Kg procesados por día y por equipo (con fila TOTAL)")
        contrib_rows = []
        for (t, f, m, q) in res.get("production", []):
            contrib_rows.append({
                "FECHA": t_to_fecha[int(t)],
                "Equipo": str(m),
                "kg": float(q) * 1000.0,
            })
        contrib_df = pd.DataFrame(contrib_rows)

        if contrib_df.empty:
            st.info("No hay kilos asignados por equipo para esta corrida.")
        else:
            pivot_equipo = (
                contrib_df.pivot_table(index="FECHA", columns="Equipo", values="kg", aggfunc="sum")
                .reindex(columns=(S_list + H_list), fill_value=0.0)
                .fillna(0.0)
                .sort_index()
            )
            total_row = pd.DataFrame(pivot_equipo.sum(axis=0)).T
            total_row.index = ["TOTAL"]
            pivot_con_total = pd.concat([pivot_equipo, total_row], axis=0)

            st.dataframe(pivot_con_total, use_container_width=True)
            st.download_button(
                "Descargar tabla (CSV)",
                data=pivot_con_total.to_csv().encode("utf-8"),
                file_name="kg_por_dia_y_equipo_con_totales.csv",
                mime="text/csv",
            )

        # =======================
        # Uso de cadena (si aplica)
        # =======================
        chain_rows = [{"FECHA": t_to_fecha[int(t)], "CAMPO": f} for (t, f) in res.get("chain_use", [])]
        chain_df = pd.DataFrame(chain_rows)
        st.subheader("⛓️ Uso de la Cadena (y[f,t]=1)")
        if chain_df.empty:
            st.info("Cadena no utilizada.")
        else:
            st.dataframe(chain_df.sort_values(["FECHA", "CAMPO"]).reset_index(drop=True), use_container_width=True)

        # =======================
        # Descargas
        # =======================
        st.subheader("📥 Descargar resultados")

        st.download_button(
            "Descargar z (CSV)",
            data=z_df.to_csv(index=False).encode("utf-8"),
            file_name="z_por_fecha_y_campo.csv",
            mime="text/csv",
        )

        st.download_button(
            "Descargar cronograma (CSV)",
            data=schedule_df.to_csv(index=False).encode("utf-8"),
            file_name="cronograma_equipos.csv",
            mime="text/csv",
        )

        st.download_button(
            "Descargar cronograma (pivot) (CSV)",
            data=pivot_df.to_csv().encode("utf-8"),
            file_name="cronograma_equipos_pivot.csv",
            mime="text/csv",
        )

        st.download_button(
            "Descargar kg por equipo (CSV)",
            data=contrib_df.to_csv(index=False).encode("utf-8"),
            file_name="kg_por_equipo_y_dia.csv",
            mime="text/csv",
        )

        sol = {
            "estado": status,
            "objetivo": obj,
            "revenue": res.get("revenue", 0.0),
            "cost_init": res.get("cost_init", 0.0),
            "cost_move": res.get("cost_move", 0.0),
            "assignments": [
                {"t": int(t), "fecha": t_to_fecha[int(t)], "team": m, "campo": f}
                for (t, m, f) in res.get("assignments", [])
            ],
            "chain_use": [
                {"t": int(t), "fecha": t_to_fecha[int(t)], "campo": f}
                for (t, f) in res.get("chain_use", [])
            ],
            "z": z_df.to_dict(orient="records"),
        }
        st.download_button(
            "Descargar JSON de la solución",
            data=json.dumps(sol, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="solucion_asignacion.json",
            mime="application/json",
        )
        st.session_state.solution   = res
        st.session_state.z_df       = z_df
        st.session_state.schedule_df= schedule_df
        st.session_state.contrib_df = contrib_df
        st.session_state.chain_df   = chain_df
        st.session_state.meta = {
            "t_to_fecha": t_to_fecha,
            "S_list": S_list,
            "H_list": H_list,
            "team_type": team_type,}

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
        st.exception(e)

# =======================
# 🔁 Render persistente
# =======================
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
