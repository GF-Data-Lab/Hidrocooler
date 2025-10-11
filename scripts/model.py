# -*- coding: utf-8 -*-
from __future__ import annotations
import os, io, json, zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, NamedTuple
import random
from itertools import product

def _read_json_from_dir(dirname: str, name: str, required=True):
    path = os.path.join(dirname, name)
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Falta '{name}' en {dirname}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ====== Tipos del modelo (igual a tu archivo de CPLEX) ======
Field = str
Team  = str
Day   = int

@dataclass
class Instance:
    fields: List[Field]
    teams: List[Team]
    days: List[Day]
    capacity_tpd: Dict[Team, float]
    start_field: Dict[Team, Field]
    move_cost: Dict[Tuple[Field, Field], float]
    revenue_per_ton: float = 9000.0
    demand_ton: Optional[Dict[Tuple[Field, Day], float]] = None
    max_teams_per_field: Optional[Dict[Tuple[Field, Day], int]] = None
    instalation_cost: float = 40000.0
    team_type: Dict[Team, str] = None
    field_allow: Dict[Field, Dict[str, bool]] = None
    max_moves_per_day: Optional[int] = None

class FixedUnit(NamedTuple):
    team: str
    field: str
    kind: str        # "Hidro" o "Sadema"
    capacity: float

def build_and_solve_docplex(inst: Instance, write_lp: bool = False, fixed_units: List[FixedUnit] = None):
    if fixed_units is None:
        fixed_units = []

    # Copias locales para no mutar el Instance original
    F = list(inst.fields)
    M = list(inst.teams)
    T = list(inst.days)

    cap = dict(inst.capacity_tpd)
    f0 = dict(inst.start_field)
    c = dict(inst.move_cost)
    c_inst = inst.instalation_cost
    r = inst.revenue_per_ton
    L = getattr(inst, "max_moves_per_day", None)

    # Copias mutables y extendidas con equipos fijos
    field_allow = {f: inst.field_allow[f].copy() for f in F}
    team_type   = dict(inst.team_type)

    # ===== Añadir equipos fijos como "extra" a M =====
    for fu in fixed_units:
        if fu.kind not in ("Hidro", "Sadema"):
            raise ValueError("kind debe ser 'Hidro' o 'Sadema'")

        if fu.team in team_type:
            raise ValueError(f"Equipo fijo '{fu.team}' ya existe en M/team_type")

        # Si el campo no permite el tipo, habilítalo explícitamente en la copia local

        # Agregar a los conjuntos y parámetros locales
        M.append(fu.team)
        team_type[fu.team] = fu.kind
        cap[fu.team] = float(fu.capacity)
        f0[fu.team]  = fu.field  # parte allí

    # ===== Validaciones básicas usando copias extendidas =====
    for m in M:
        assert m in cap, f"Falta capacidad para equipo {m}"
        assert m in f0, f"Falta campo inicial para equipo {m}"
        assert f0[m] in F, f"Campo inicial {f0[m]} de {m} no está en la lista de campos"
        assert team_type is not None and m in team_type, f"Falta tipo (Sadema/Hidro) para equipo {m}"
        assert team_type[m] in ("Sadema", "Hidro"), f"Tipo inválido para {m}: {team_type[m]}"

    for f in F:
        assert field_allow is not None and f in field_allow, f"Faltan flags de admisión para campo {f}"
        for key in ("Sadema", "Hidro", "Chain"):
            assert key in field_allow[f], f"Falta flag '{key}' en field_allow[{f}]"

    # Partición por tipo a partir del diccionario extendido
    S = [m for m in M if team_type[m] == "Sadema"]
    H = [m for m in M if team_type[m] == "Hidro"]

    # ===== Parámetros de viaje con retardos δ =====
    def travel_time(f, f2):
        return 1 if f == f2 else 2

    idx = {t: i for i, t in enumerate(T)}
    nT = len(T)

    from docplex.mp.model import Model
    mdl = Model(name="Asignacion_Sadema_Hidromovil_Cadena_delta")

    # ===== Variables =====
    # Dominios habilitados por field_allow (copia local actualizada)
    x_keys = []
    F_H = [f for f in F if field_allow[f]["Hidro"]]
    F_S_plain = [f for f in F if field_allow[f]["Sadema"]]
    F_S_chain = [f for f in F if field_allow[f]["Chain"] and not field_allow[f]["Sadema"]]

    for m in H:
        for f in F_H:
            for t in T:
                x_keys.append((m, f, t))
    for m in S:
        for f in (F_S_plain + F_S_chain):
            for t in T:
                x_keys.append((m, f, t))

    x = mdl.binary_var_dict(x_keys, name="x")  # 1 si m opera en f el día t
    # tras crear x_keys con móviles:
    for fu in fixed_units:
        for t in T:
            if (fu.team, fu.field, t) not in x_keys:
                x_keys.append((fu.team, fu.field, t))

    p = mdl.continuous_var_dict([(m, f, t) for m in M for f in F for t in T], lb=0, name="p")  # ton procesadas
    y = mdl.binary_var_dict([(f, t) for f in F for t in T], name="y")  # cadena en f,t

    # Variables de movimiento con retardos: solo si t+δ dentro del horizonte
    w_keys = []
    for m, f, f2, t in product(M, F, F, T):
        delta = travel_time(f, f2)
        if idx[t] + delta < nT:
            w_keys.append((m, f, f2, t))
    w = mdl.binary_var_dict(w_keys, name="w")

    # Helper para x inexistente -> 0 (como tu X(m,f,t))
    def X(m, f, t):
        # Devuelve la variable si existe; si no, un literal 0
        return x[(m, f, t)] if (m, f, t) in x else 0

    # --- Pin equipos fijos a su campo todo el horizonte ---
    t0 = T[0]
    for fu in fixed_units:
        m = fu.team
        fstar = fu.field
        # fuerza x=1 en fstar para todo t, y 0 en otros campos
        for t in T:
            if (m, fstar, t) in x:
                mdl.add_constraint(x[(m, fstar, t)] == 1, ctname=f"fix_{m}_{fstar}_{t}")
            for f in F:
                if f != fstar and (m, f, t) in x:
                    mdl.add_constraint(x[(m, f, t)] == 0, ctname=f"nofix_{m}_{f}_{t}")

        # bloquea movimientos (solo “quedarse”)
        # w(m, fstar, fstar, t) debe existir y valer 1 para t con t+1 dentro del horizonte;
        # todos los w con f!=fstar o f2!=fstar los ponemos 0.
        for (mm, f, f2, tt), var in w.items():
            if mm != m:
                continue
            if f == fstar and f2 == fstar:
                mdl.add_constraint(var == 1, ctname=f"stay_{m}_{tt}")
            else:
                mdl.add_constraint(var == 0, ctname=f"nomove_{m}_{f}_{f2}_{tt}")

    # ===== Objetivo =====
    revenue = mdl.sum(r * p[m, f, t] for m, f, t in product(M, F, T))

    cost_init = mdl.sum((c.get((f0[m], f), 0.0)) * X(m, f, T[0]) for m in M for f in F)
    cost_move = mdl.sum(c.get((f, f2), 0.0) * w[m, f, f2, t] for (m, f, f2, t) in w_keys)

    cost_inst_init = mdl.sum(c_inst * X(m, f, T[0]) for m in M for f in F if f != f0[m])
    cost_inst_move = mdl.sum(c_inst * w[m, f, f2, t] for (m, f, f2, t) in w_keys if f != f2)

    mdl.maximize(revenue - cost_init - cost_move - cost_inst_init - cost_inst_move)

    # ===== Restricciones =====
    # (1) Asignación diaria: a lo más un campo por equipo y día
    for m, t in product(M, T):
        mdl.add_constraint(mdl.sum(X(m, f, t) for f in F) <= 1, ctname=f"asign_atmost1_{m}_{t}")

    # (2) Capacidad
    for m, f, t in product(M, F, T):
        mdl.add_constraint(p[m, f, t] <= cap[m] * X(m, f, t), ctname=f"cap_{m}_{f}_{t}")

    # (3) Demanda por campo y día (opcional)
    if inst.demand_ton is not None:
        for f, t in product(F, T):
            if (f, t) in inst.demand_ton:
                mdl.add_constraint(mdl.sum(p[m, f, t] for m in M) <= inst.demand_ton[(f, t)], ctname=f"dem_{f}_{t}")

    # (3b) Máximo de equipos por campo y día (opcional)
    fixed_set = set(fu.team for fu in fixed_units)  # nombres de equipos fijos
    for f, t in product(F, T):
        if (f, t) in inst.max_teams_per_field:
            mdl.add_constraint(
                mdl.sum(X(m, f, t) for m in M if m not in fixed_set) 
                <= 1,
                ctname=f"maxK_moviles_{f}_{t}"
            )

    # (4) Compatibilidades (usar field_allow actualizado)
    # Hidro solo en campos permitidos
    for m, f, t in product(H, F, T):
        if not field_allow[f]["Hidro"]:
            if (m, f, t) in x:
                mdl.add_constraint(X(m, f, t) == 0, ctname=f"compat_H_{m}_{f}_{t}")

    # Sadema: si no permite Sadema pero sí Chain => requiere y[f,t]; si no permite nada => 0
    fixed_set = set(fu.team for fu in fixed_units)

    for m, f, t in product(S, F, T):
        allow_S = field_allow[f]["Sadema"]
        allow_chain = field_allow[f]["Chain"]
        if m in fixed_set:
            # El fijo opera “aparte”: lo fijas en su campo y 0 en los demás; no necesita y.
            continue
        if not allow_S:
            if allow_chain:
                mdl.add_constraint(X(m, f, t) <= y[f, t], ctname=f"compat_S_chain_MOVIL_{m}_{f}_{t}")
            else:
                if (m, f, t) in x:
                    mdl.add_constraint(X(m, f, t) == 0, ctname=f"compat_S_forbidden_MOVIL_{m}_{f}_{t}")



    # (5) Recurso Cadena
    for t in T:
        mdl.add_constraint(mdl.sum(y[f, t] for f in F) <= 1, ctname=f"cadena_cap_{t}")

    for f, t in product(F, T):
        # y solo permitido donde Chain=True y Sadema=False
        if (not field_allow[f]["Chain"]) or field_allow[f]["Sadema"]:
            mdl.add_constraint(y[f, t] == 0, ctname=f"cadena_only_where_allowed_{f}_{t}")

    for f, t in product(F, T):
        if field_allow[f]["Chain"] and not field_allow[f]["Sadema"]:
            # Si hay Sadema operando ahí, obliga y=1 (sum X <= y)
            mdl.add_constraint(mdl.sum(X(m, f, t) for m in S) <= y[f, t], ctname=f"one_sadema_with_chain_{f}_{t}")

    # (6) Flujo con retardos δ
    # (6a) Salida: suma de arcos saliendo desde (m,f,t) = x(m,f,t)
    for m, f, t in product(M, F, T):
        arcs_out = [(m, f, f2, t) for f2 in F if (m, f, f2, t) in w]
        if arcs_out:
            mdl.add_constraint(mdl.sum(w[k] for k in arcs_out) == X(m, f, t), ctname=f"salida_{m}_{f}_{t}")

    # (6b) Entrada: suma de predecesores con retardo = x(m,f2,t2)
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
                key = (m, f, f2, t_prev)
                if key in w:
                    preds.append(w[key])
        if preds:
            mdl.add_constraint(mdl.sum(preds) == X(m, f2, t2), ctname=f"entrada_{m}_{f2}_{t2}")
        else:
            # Si no hay predecesores válidos, entonces x debe ser 0 (consistencia)
            if (m, f2, t2) in x:
                mdl.add_constraint(X(m, f2, t2) == 0, ctname=f"entrada_empty_{m}_{f2}_{t2}")

    # (6c) Máximo de movimientos diarios global (opcional)
    if L is not None:
        for t in T:
            arcs_t = [(m, f, f2, tt) for (m, f, f2, tt) in w_keys if tt == t and f2 != f]
            if arcs_t:
                mdl.add_constraint(mdl.sum(w[k] for k in arcs_t) <= L, ctname=f"max_moves_day_{t}")

    # Export LP (opcional)
    if write_lp:
        mdl.export_as_lp("modelo_asignacion_cadena_con_deltas.lp")

    # ===== Parámetros de resolución =====
    mdl.parameters.mip.tolerances.mipgap = 0.01
    mdl.parameters.threads = 90
    mdl.set_log_output(True)

    sol = mdl.solve(log_output=True)

    if not sol:
        cf = mdl.refine_conflict()
        cf.display()  # imprime las restricciones en conflicto

    # ===== Resultados =====
    res = {
        "status": mdl.solve_details.status,   # string
        "objective": None,
        "revenue": None,
        "cost_init": None,
        "cost_move": None,
        "assignments": [],
        "production": [],
        "chain_use": [],
    }

    if not sol:
        return res

    # Evalúa términos del objetivo para desglose
    revenue_val = sol.get_value(revenue)
    cost_init_val = sol.get_value(cost_init)
    cost_move_val = sol.get_value(cost_move)
    obj_val = mdl.objective_value

    res.update({
        "objective": obj_val,
        "revenue": revenue_val,
        "cost_init": cost_init_val,
        "cost_move": cost_move_val,
    })

    # Asignaciones, producción y uso de cadena
    for t in T:
        for m in M:
            for f in F:
                xv = sol.get_value(x[(m, f, t)]) if (m, f, t) in x else 0.0
                if xv is not None and xv > 0.5:
                    res["assignments"].append((t, m, f))
                pv = sol.get_value(p[m, f, t])
                if pv is not None and pv > 1e-6:
                    res["production"].append((t, f, m, pv))
        for f in F:
            yv = sol.get_value(y[f, t])
            if yv is not None and yv > 0.5:
                res["chain_use"].append((t, f))

    return res

def make_instance(seed: int = 42) -> Instance:
    random.seed(seed)
    dir = "./"
    #EQUIPOS
    json_teams = _read_json_from_dir(dir, 'teams.json')
    json_demand = _read_json_from_dir(dir,'demand.json')
    json_costs = _read_json_from_dir(dir,'costs.json')
    json_fields_flags = _read_json_from_dir(dir,'fields_flags.json')
    json_params = _read_json_from_dir(dir,'params.json')
    json_calendar = _read_json_from_dir(dir, 'calendar.json')
    # Conjuntos
    T = list(set([i['t'] for i in json_calendar]))

    F = list(set([i['campo'] for i in json_demand]))
    S_teams = [f"{i['equipo']}" for i in json_teams if i['tipo']=='Sadema']
    H_teams = [f"{i['equipo']}" for i in json_teams if i['tipo']=='Hidro']
    M = S_teams + H_teams

    # Tipos
    team_type = {m:"Sadema" for m in S_teams}
    team_type.update({m:"Hidro" for m in H_teams})

    # Capacidades
    capacity_tpd = {m:(json_params['gamma_S_default'] if m.startswith("S") else json_params['gamma_H_default']) for m in M}


    field_allow = {}
    for f in json_fields_flags:
        field_allow[f['campo']] = {}
        field_allow[f['campo']]['Sadema'] = f['Sadema']
        field_allow[f['campo']]['Hidro'] = f['Hidro']
        field_allow[f['campo']]['Chain'] = f['Chain']
    # Campo inicial (distribuidos)
    start_field = {}
    for i in json_teams:
        start_field[i['equipo']]=i['campo_inicial']

    # Costos de movimiento: simétricos, cero en diagonal, 1$/ton·km * distancia “anillo”
    # (usa distancia circular para tener estructura suave)

    move_cost = {}
    for i in json_costs:
        move_cost[(i['origen'], i['destino'])] = i['costo']

    # Demanda diaria (ton): patrón suave + ruido, 0–150 ton/día/campo máx
    demand_ton = {}
    for i in json_demand:
        demand_ton[(i['campo'], i['t'])] = i['demanda_ton']
    # Límite de equipos por campo/día (opcional): p.ej. 2
    max_teams_per_field = {(f,t): 99999 for f in F for t in T}

    return Instance(
        fields=F, teams=M, days=T,
        capacity_tpd=capacity_tpd,
        start_field=start_field,
        move_cost=move_cost,
        revenue_per_ton=json_params['ganancia_ton'],
        demand_ton=demand_ton,
        max_teams_per_field=max_teams_per_field,
        instalation_cost=json_params['instalation_cost'],
        team_type=team_type,
        field_allow=field_allow,
        max_moves_per_day=json_params['max_moves_per_day'],   # puedes bajar/subir este L
    )

inst = make_instance()
test_fijos = _read_json_from_dir("./",'fields_flags.json')
json_params = _read_json_from_dir("./", 'params.json')
fixed = []
h_fix = 1
s_fix = 1
for i in test_fijos:
    if i['AdemaFija'] == True:
        fixed.append(FixedUnit(team=f"Sfix{s_fix}", field=i['campo'], kind="Sadema",  capacity=json_params['gamma_S_default']))
        s_fix+=1
    if i['HidroFijo'] == True:
        fixed.append(FixedUnit(team=f"Hfix{h_fix}", field=i['campo'], kind="Hidro",  capacity=json_params['gamma_H_default']))
        h_fix+=1

res = build_and_solve_docplex(inst=inst, fixed_units=fixed)