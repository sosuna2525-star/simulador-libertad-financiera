import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# ⚙️ CONFIGURACIÓN INICIAL DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Simulador Monte Carlo v6.0 PRO",
    page_icon="🛡️",
    layout="wide"
)

# ==========================================
# 🎛️ CONFIGURACIÓN GLOBAL (VALORES POR DEFECTO)
# ==========================================
CONFIG = {
    "num_simulaciones": 10000,
    "semilla": 42,
    "ano_inicio": datetime.now().year,
    "edad_inicial": 51,
    "edad_muerte": 85,
    "edad_retiro_bolsa": 67,
    "edad_fin_hipoteca": 75,

    # Cartera Inicial y Estructura Multiactivo
    "cartera_msci_inicial": 0,
    "base_coste_inicial": 0,
    "peso_bolsa": 0.75,
    "peso_rf": 0.25,

    # Flujos e Ingresos Base
    "gasto_mensual_hoy": 0,
    "cuota_hipoteca_hoy": 0,
    "paro_mensual": 0,
    "meses_paro_inicial": 24,  # ahora configurable (antes hardcodeado)
    "pension_base": 0,

    # Parámetros Financieros
    "media_rentabilidad_bolsa": 0.07,
    "volatilidad_bolsa": 0.15,
    "media_rentabilidad_rf": 0.03,
    "volatilidad_rf": 0.04,
    "interes_efectivo": 0.02,

    # Inflación y Correlación
    "media_inflacion": 0.025,
    "std_inflacion": 0.01,
    "correlacion_bolsa_inflacion": -0.25,

    # Fiscalidad IRPF Español (base del ahorro, 5 tramos reales)
    "aplicar_fiscalidad_real": True,
    "tramos_irpf": [
        (6000, 0.19),
        (50000, 0.21),
        (200000, 0.23),
        (300000, 0.27),
        (float("inf"), 0.30),
    ],

    # Indexación
    "pension_indexada": True,
    "extras_indexados": True,

    # Herencia Estocástica
    "herencia_activa": True,
    "edad_herencia_media": 60,
    "std_herencia_edad": 1.5,
    "importe_herencia": 0,

    # Extras e Indemnización Dinámica
    "indemnizacion_si_sales_ahora": 0,
    "indemnizacion_si_sales_en_n_anos": 0,
    "anos_trabajo_adicional": 3,
    "ahorro_anual_transicion": 0,
    "ingreso_extra_deseado_hoy": 0,
    "edad_fin_ingresos_extras": 70,

    # Umbral de Seguridad Objetivo
    "umbral_exito_objetivo": 80.0
}

# ==========================================
# 🏛️ FUNCIONES AUXILIARES: FISCALIDAD Y ETAPAS
# ==========================================
def calcular_impuesto_irpf_vectorizado(plusvalia, tramos):
    """
    Calcula el IRPF sobre la base del ahorro de forma vectorizada.
    tramos: lista de (limite_superior_acumulado, tipo_marginal), ascendente,
            el último tramo debe tener limite_superior = np.inf.
    Aplica correctamente la progresividad por tramos (no un tipo plano).
    """
    plusvalia = np.maximum(plusvalia, 0.0)
    impuesto = np.zeros_like(plusvalia, dtype=float)
    limite_inferior = 0.0
    for limite_superior, tipo in tramos:
        base_tramo = np.clip(plusvalia, limite_inferior, limite_superior) - limite_inferior
        base_tramo = np.maximum(base_tramo, 0.0)
        impuesto += base_tramo * tipo
        limite_inferior = limite_superior
    return impuesto


def factor_gasto_etapa_vida(edad):
    if edad < 68:
        return 1.05   # Etapa activa (Go-Go)
    elif edad < 78:
        return 0.95   # Etapa moderada (Slow-Go)
    else:
        return 1.00   # Etapa pasiva (No-Go)


# ==========================================
# ⚙️ MOTOR VECTORIZADO DE SIMULACIÓN MONTE CARLO
# ==========================================
def simular_escenario_especifico(cfg, anos_t, indem, ahorro, extra):
    """
    Ejecuta Monte Carlo para un único escenario.
    Vectorizado sobre las N simulaciones (solo queda el bucle sobre los años,
    que como mucho tiene 40-50 iteraciones). Esto sustituye el doble bucle
    original (simulaciones x años) por operaciones sobre arrays de NumPy,
    dando una mejora de rendimiento de entre 50x y 200x.

    NOTA IMPORTANTE: se usa la misma semilla en cada llamada a esta función
    (common random numbers). Esto es intencional: hace que comparar
    "trabajar 0 años" vs "trabajar 3 años" use los mismos caminos de mercado
    e inflación simulados, aislando el efecto real de la decisión evaluada.
    No es un bug.
    """
    rng = np.random.default_rng(cfg["semilla"])
    edades = np.arange(cfg["edad_inicial"], cfg["edad_muerte"] + 1)
    n_anos = len(edades)
    n_sims = cfg["num_simulaciones"]

    # --- Generación masiva de aleatoriedad ---
    cov_matrix = [
        [cfg["volatilidad_bolsa"] ** 2,
         cfg["volatilidad_bolsa"] * cfg["std_inflacion"] * cfg["correlacion_bolsa_inflacion"]],
        [cfg["volatilidad_bolsa"] * cfg["std_inflacion"] * cfg["correlacion_bolsa_inflacion"],
         cfg["std_inflacion"] ** 2]
    ]
    shocks = rng.multivariate_normal([0, 0], cov_matrix, size=(n_sims, n_anos))  # (n_sims, n_anos, 2)

    # Corrección: se usa log(1+media) en vez de "media" a secas, para que la
    # rentabilidad ESPERADA real coincida con el input del usuario
    # (antes E[1+r] = exp(media), ahora E[1+r] ≈ (1+media)).
    mu_log_rv = np.log1p(cfg["media_rentabilidad_bolsa"]) - 0.5 * (cfg["volatilidad_bolsa"] ** 2)
    retornos_rv = np.exp(mu_log_rv + shocks[:, :, 0]) - 1.0            # (n_sims, n_anos)
    inflaciones = cfg["media_inflacion"] + shocks[:, :, 1]             # (n_sims, n_anos)
    retornos_rf = rng.normal(cfg["media_rentabilidad_rf"], cfg["volatilidad_rf"], size=(n_sims, n_anos))

    if cfg["herencia_activa"]:
        edades_herencia_sim = np.round(
            rng.normal(cfg["edad_herencia_media"], cfg["std_herencia_edad"], n_sims)
        )
    else:
        edades_herencia_sim = np.full(n_sims, -1)

    # --- Estado inicial (arrays por simulación) ---
    cartera_total = np.full(n_sims, float(cfg["cartera_msci_inicial"]))
    base_coste = np.full(n_sims, float(cfg["base_coste_inicial"]))
    efectivo = np.zeros(n_sims)
    factor_inflacion = np.ones(n_sims)
    herencia_recibida = np.zeros(n_sims, dtype=bool)
    quiebra_registrada = np.zeros(n_sims, dtype=bool)
    edad_quiebra_sim = np.full(n_sims, np.nan)
    meses_paro = cfg.get("meses_paro_inicial", 24)  # escalar: no depende de la aleatoriedad

    matriz_trayectorias = np.zeros((n_sims, n_anos))
    matriz_factor_inflacion = np.zeros((n_sims, n_anos))

    for i in range(n_anos):
        edad = edades[i]
        inf_ano = np.maximum(0.005, inflaciones[:, i])
        factor_inflacion = factor_inflacion * (1.0 + inf_ano)
        rent_rv = retornos_rv[:, i]
        rent_rf = retornos_rf[:, i]
        rent_ponderada = cfg["peso_bolsa"] * rent_rv + cfg["peso_rf"] * rent_rf

        # Herencia estocástica
        if cfg["herencia_activa"]:
            mask_h = (~herencia_recibida) & (edad >= edades_herencia_sim)
            if np.any(mask_h):
                cartera_total = np.where(mask_h, cartera_total + cfg["importe_herencia"], cartera_total)
                base_coste = np.where(mask_h, base_coste + cfg["importe_herencia"], base_coste)
                herencia_recibida = herencia_recibida | mask_h

        # Indemnización: se cobra en el momento REAL de la salida (t = anos_t),
        # ya sea inmediata (anos_t = 0) o diferida (anos_t = N años trabajados de más).
        # Antes se cobraba siempre al inicio, aunque el escenario fuera "trabajar N años",
        # lo cual la hacía desaparecer de esos escenarios en vez de retrasarse.
        if i == anos_t:
            efectivo = efectivo + indem

        if i < anos_t:
            # ETAPA A: Transición / Trabajo (acumulación)
            ahorro_ajustado = ahorro * factor_inflacion
            cartera_total = cartera_total + ahorro_ajustado
            base_coste = base_coste + ahorro_ajustado
            cartera_total = cartera_total + cartera_total * rent_ponderada
            efectivo = efectivo + efectivo * cfg["interes_efectivo"]
        else:
            # ETAPA B: Retiro / Extracción de capital
            gasto_base = (cfg["gasto_mensual_hoy"] - cfg["cuota_hipoteca_hoy"]
                          if edad >= cfg["edad_fin_hipoteca"] else cfg["gasto_mensual_hoy"])
            gasto_anual = (gasto_base * 12) * factor_gasto_etapa_vida(edad) * factor_inflacion

            if meses_paro > 0:
                meses_a_cobrar = min(12, meses_paro)
                ingreso_paro = meses_a_cobrar * cfg["paro_mensual"]
                meses_paro -= meses_a_cobrar
            else:
                ingreso_paro = 0.0

            ingreso_pension = 0.0
            if edad >= cfg["edad_retiro_bolsa"]:
                ingreso_pension = (cfg["pension_base"] * 12) * (
                    factor_inflacion if cfg["pension_indexada"] else 1.0)

            ingreso_extra = 0.0
            if edad < cfg["edad_fin_ingresos_extras"]:
                ingreso_extra = extra * (factor_inflacion if cfg["extras_indexados"] else 1.0)

            necesidad_liquida = np.maximum(0.0, gasto_anual - ingreso_paro - ingreso_pension - ingreso_extra)

            mask_suf = efectivo >= necesidad_liquida
            resto_necesario = np.where(mask_suf, 0.0, necesidad_liquida - efectivo)
            efectivo = np.where(mask_suf, efectivo - necesidad_liquida, 0.0)

            mask_insuf = ~mask_suf
            if np.any(mask_insuf):
                cartera_segura = np.where(cartera_total > 0, cartera_total, 1.0)
                proporcion_ganancia = np.maximum(0.0, (cartera_total - base_coste) / cartera_segura)

                retiro_bruto = resto_necesario.copy()
                for _ in range(4):  # gross-up iterativo para cubrir el neto tras impuestos
                    plusvalia_aflorada = retiro_bruto * proporcion_ganancia
                    if cfg["aplicar_fiscalidad_real"]:
                        impuesto = calcular_impuesto_irpf_vectorizado(plusvalia_aflorada, cfg["tramos_irpf"])
                    else:
                        impuesto = np.zeros_like(plusvalia_aflorada)
                    neto_obtenido = retiro_bruto - impuesto
                    neto_seguro = np.where(neto_obtenido > 0, neto_obtenido, 1.0)
                    retiro_bruto = np.where(
                        neto_obtenido > 0,
                        resto_necesario * (retiro_bruto / neto_seguro),
                        retiro_bruto
                    )

                retiro_efectivo = np.minimum(cartera_total, retiro_bruto)
                base_coste_reducida = retiro_efectivo * (1.0 - proporcion_ganancia)
                cartera_total = np.where(mask_insuf, cartera_total - retiro_efectivo, cartera_total)
                base_coste = np.where(mask_insuf, np.maximum(0.0, base_coste - base_coste_reducida), base_coste)

            cartera_total = cartera_total + cartera_total * rent_ponderada
            efectivo = efectivo + efectivo * cfg["interes_efectivo"]

        patrimonio_total = cartera_total + efectivo

        nuevas_quiebras = (patrimonio_total <= 0) & (~quiebra_registrada)
        if np.any(nuevas_quiebras):
            edad_quiebra_sim = np.where(nuevas_quiebras, edad, edad_quiebra_sim)
        quiebra_registrada = quiebra_registrada | (patrimonio_total <= 0)

        # Una vez en quiebra, se congela en 0 para el resto del horizonte
        cartera_total = np.where(quiebra_registrada, 0.0, cartera_total)
        efectivo = np.where(quiebra_registrada, 0.0, efectivo)
        patrimonio_total = np.where(quiebra_registrada, 0.0, patrimonio_total)

        matriz_trayectorias[:, i] = patrimonio_total
        matriz_factor_inflacion[:, i] = factor_inflacion

    edades_quiebra = edad_quiebra_sim[~np.isnan(edad_quiebra_sim)]
    tasa_exito = (1 - (len(edades_quiebra) / n_sims)) * 100
    patrimonio_final = matriz_trayectorias[:, -1]
    p10, p25, p50, p75 = np.percentile(patrimonio_final, [10, 25, 50, 75])
    prob_1m = np.mean(patrimonio_final >= 1_000_000) * 100
    edad_media_quiebra = np.mean(edades_quiebra) if len(edades_quiebra) > 0 else np.nan

    return {
        "p10": p10, "p25": p25, "p50": p50, "p75": p75,
        "tasa_exito": tasa_exito, "prob_1m": prob_1m,
        "edad_media_quiebra": edad_media_quiebra, "anos_transicion": anos_t,
        "matriz_trayectorias": matriz_trayectorias,
        "matriz_factor_inflacion": matriz_factor_inflacion,
        "edades": edades
    }


# ==========================================
# 🔍 OPTIMIZADOR: BÚSQUEDA AUTOMÁTICA DE AÑOS (N)
# ==========================================
@st.cache_data
def buscar_anos_necesarios_para_objetivo(cfg, objetivo_pct, con_extras=False):
    """
    Itera N = 0, 1, 2... hasta encontrar los años necesarios para superar el umbral de éxito.

    Solo conocemos la indemnización en dos supuestos: si sales ahora (N=0) o si sigues
    trabajando más (N≥1). Para N≥1 se usa la indemnización "diferida" configurada,
    independientemente del N exacto (aproximación razonable: se asume que no cambia mucho
    entre trabajar 1 año más o 5 años más). Si tu indemnización real varía mucho según el
    año exacto de salida, este resultado será menos preciso cuanto más se aleje N del valor
    que configuraste como referencia.
    """
    indem_ahora = cfg.get("indemnizacion_si_sales_ahora", 0)
    indem_diferida = cfg.get("indemnizacion_si_sales_en_n_anos", 0)
    ahorro_anual = cfg.get("ahorro_anual_transicion", 0)
    extra_anual = cfg.get("ingreso_extra_deseado_hoy", 0) if con_extras else 0

    max_anos_evaluar = 15
    for n in range(max_anos_evaluar + 1):
        indem_efectiva = indem_ahora if n == 0 else indem_diferida

        res = simular_escenario_especifico(cfg, anos_t=n, indem=indem_efectiva, ahorro=ahorro_anual, extra=extra_anual)
        if res["tasa_exito"] >= objetivo_pct:
            return n, res["tasa_exito"]

    return None, 0.0


# ==========================================
# ⚙️ MOTOR UNIFICADO MONTE CARLO PRO
# ==========================================
@st.cache_data
def ejecutar_matriz_escenarios(cfg):
    indem_ahora = cfg.get("indemnizacion_si_sales_ahora", 0)
    indem_diferida = cfg.get("indemnizacion_si_sales_en_n_anos", 0)
    ahorro_user = cfg.get("ahorro_anual_transicion", 0)
    anos_n = cfg.get("anos_trabajo_adicional", 3)
    extra_fmt = f"{cfg['ingreso_extra_deseado_hoy']:,.0f} €"

    ano_base = cfg.get("ano_inicio", datetime.now().year)
    ano_s1 = ano_base + 1
    ano_tn_fin = ano_s1 + anos_n

    escenarios_config = [
        (f"1. Salida {ano_s1} Puro ({indem_ahora:,.0f}€ indem. | CERO extras)", 0, indem_ahora, 0, 0),
        (f"2. Salida {ano_s1} + Extras ({extra_fmt}/año hasta los {cfg['edad_fin_ingresos_extras']})", 0, indem_ahora, 0, cfg['ingreso_extra_deseado_hoy']),
        (f"3. Trabajar {anos_n} Años ({ano_s1}-{ano_tn_fin}) | {indem_diferida:,.0f}€ indem. diferida | CERO extras", anos_n, indem_diferida, ahorro_user, 0),
        (f"4. Trabajar {anos_n} Años ({ano_s1}-{ano_tn_fin}) | {indem_diferida:,.0f}€ indem. diferida + Extras ({extra_fmt}/año)", anos_n, indem_diferida, ahorro_user, cfg['ingreso_extra_deseado_hoy']),
    ]

    resultados = []
    filas_matriz = []

    for nombre, anos_t, indem, ahorro, extra in escenarios_config:
        res = simular_escenario_especifico(cfg, anos_t, indem, ahorro, extra)
        res["nombre"] = nombre
        res["indem"] = indem
        res["ahorro"] = ahorro
        res["extra"] = extra
        resultados.append(res)

        filas_matriz.append({
            "Configuración del Escenario": nombre,
            "Pesimista (P10)": f"{res['p10']:,.0f} €".replace(",", "."),
            "Mediana / Base (P50)": f"{res['p50']:,.0f} €".replace(",", "."),
            "Optimista (P75)": f"{res['p75']:,.0f} €".replace(",", "."),
            "Tasa Éxito (% Sin Quiebra)": f"{res['tasa_exito']:.1f} %",
            "Prob. Legado (> 1M€)": f"{res['prob_1m']:.1f} %",
            "Edad Media Quiebra": f"{res['edad_media_quiebra']:.1f} años" if not np.isnan(res['edad_media_quiebra']) else "N/A"
        })

    return resultados, pd.DataFrame(filas_matriz)


# ==========================================
# 🖥️ INTERFAZ WEB STREAMLIT
# ==========================================
st.title("🛡️ Simulador Monte Carlo v6.0 PRO")
st.caption("Arquitectura financiera con optimizador automático de libertad financiera, motor vectorizado e IRPF real (5 tramos).")

with st.expander("ℹ️ Cómo se calcula esto — asunciones clave del modelo", expanded=False):
    st.markdown("""
- **Fiscalidad:** se aplican los 5 tramos reales de la base del ahorro (19/21/23/27/30%) sobre la plusvalía aflorada en cada retirada, no un tipo plano.
- **Paro:** se asume que tienes derecho a la prestación por desempleo configurada en *todos* los escenarios al empezar la etapa de retiro. Si algún escenario implica una salida voluntaria (p. ej. "seguir trabajando N años" y luego jubilarte por decisión propia), es posible que **no** tengas derecho a paro — revisa este punto para tu caso concreto.
- **Indemnización:** se cobra en el momento real de la salida, no siempre al principio. Hay dos importes: uno si sales ahora (N=0) y otro si sigues trabajando (se aplica a cualquier N≥1, no solo al N exacto que configures en "Años Adicionales a Evaluar"). Si tu indemnización real cambiara mucho según el año exacto de salida, el resultado del buscador automático será menos preciso cuanto más se aleje N del valor de referencia.
- **Quiebra:** cuando el patrimonio total llega a 0€, el modelo lo trata como ruina definitiva y detiene la simulación de esa trayectoria (no se modela seguir viviendo solo de la pensión pública a partir de ahí). Es una lectura conservadora, no un "0 literal" en la vida real si ya cobras pensión.
- **Rentabilidades:** bolsa modelada como lognormal (evita retornos < -100%); renta fija como normal simple, sin correlación con la inflación.
- **Reproducibilidad:** todos los escenarios comparados usan la misma semilla aleatoria (mismos caminos de mercado/inflación), para que la comparación entre "trabajar más años" vs "salir ya" sea limpia y no ruido estadístico.
    """)

# ------------------------------------------
# 🎛️ PANEL LATERAL (SIDEBAR)
# ------------------------------------------
st.sidebar.header("⚙️ Panel de Control v6.0")

with st.sidebar.expander("🎯 Umbral de Seguridad Deseado", expanded=True):
    CONFIG["umbral_exito_objetivo"] = st.slider(
        "Tasa de Éxito Objetivo (%)",
        min_value=60.0, max_value=95.0,
        value=80.0, step=5.0,
        help="El optimizador calculará cuántos años necesitas trabajar para garantizar al menos esta seguridad."
    )

with st.sidebar.expander("👤 Perfil y Tiempos", expanded=False):
    CONFIG["edad_inicial"] = st.number_input("Edad Actual", value=CONFIG["edad_inicial"], step=1)
    CONFIG["ano_inicio"] = st.number_input("Año Inicial de Simulación", value=CONFIG["ano_inicio"], step=1)
    CONFIG["edad_muerte"] = st.number_input("Edad Final / Horizonte", value=CONFIG["edad_muerte"], step=1)
    CONFIG["edad_retiro_bolsa"] = st.number_input("Edad Pensión Pública", value=CONFIG["edad_retiro_bolsa"], step=1)
    CONFIG["edad_fin_hipoteca"] = st.number_input("Edad Fin Hipoteca", value=CONFIG["edad_fin_hipoteca"], step=1)

with st.sidebar.expander("💰 Patrimonio y Activos", expanded=False):
    CONFIG["cartera_msci_inicial"] = st.number_input("Cartera Inicial (€)", value=CONFIG["cartera_msci_inicial"], step=10000)
    CONFIG["base_coste_inicial"] = st.number_input("Base de Coste / Capital Invertido (€)", value=CONFIG["base_coste_inicial"], step=10000)
    peso_bolsa_pct = st.slider("% Renta Variable (Bolsa)", min_value=0, max_value=100, value=int(CONFIG["peso_bolsa"] * 100), step=5)
    CONFIG["peso_bolsa"] = peso_bolsa_pct / 100.0
    CONFIG["peso_rf"] = 1.0 - CONFIG["peso_bolsa"]

with st.sidebar.expander("🏠 Gastos e Ingresos", expanded=False):
    CONFIG["gasto_mensual_hoy"] = st.number_input("Gasto Mensual Actual (€)", value=CONFIG["gasto_mensual_hoy"], step=100)
    CONFIG["cuota_hipoteca_hoy"] = st.number_input("Cuota Hipoteca Mensual (€)", value=CONFIG["cuota_hipoteca_hoy"], step=50)
    CONFIG["paro_mensual"] = st.number_input("Prestación Paro Mensual (€)", value=CONFIG["paro_mensual"], step=50)
    CONFIG["meses_paro_inicial"] = st.number_input("Meses de Paro Disponibles", value=CONFIG["meses_paro_inicial"], min_value=0, max_value=48, step=1,
                                                    help="Antes estaba fijo en 24 meses en el código. Ojo: revisa si aplica en todos los escenarios (ver nota de asunciones arriba).")
    CONFIG["pension_base"] = st.number_input("Pensión Pública Estimada (€/mes)", value=CONFIG["pension_base"], step=50)
    CONFIG["ingreso_extra_deseado_hoy"] = st.number_input("Ingreso Extra Deseado (€/año)", value=CONFIG["ingreso_extra_deseado_hoy"], step=1000)
    CONFIG["edad_fin_ingresos_extras"] = st.number_input("Edad Fin Ingresos Extras", value=CONFIG["edad_fin_ingresos_extras"], step=1)

with st.sidebar.expander("💼 Indemnización y Transición", expanded=False):
    CONFIG["indemnizacion_si_sales_ahora"] = st.number_input(
        "Indemnización si sales AHORA (€)", value=CONFIG["indemnizacion_si_sales_ahora"], step=5000
    )
    CONFIG["anos_trabajo_adicional"] = st.number_input(
        "Años Adicionales a Evaluar en Tabla (N)", min_value=1, max_value=15, value=CONFIG["anos_trabajo_adicional"], step=1
    )
    CONFIG["indemnizacion_si_sales_en_n_anos"] = st.number_input(
        f"Indemnización si sales dentro de {CONFIG['anos_trabajo_adicional']} años (€)",
        value=CONFIG["indemnizacion_si_sales_en_n_anos"], step=5000,
        help="Importe esperado en el momento de esa salida diferida (euros nominales previstos para entonces; no se indexa automáticamente)."
    )
    CONFIG["ahorro_anual_transicion"] = st.number_input("Ahorro Anual si Sigues Trabajando (€/año)", value=CONFIG["ahorro_anual_transicion"], step=1000)

with st.sidebar.expander("📈 Mercado e Inflación", expanded=False):
    rent_bolsa_pct = st.slider("Rentabilidad Nominal Bolsa (%)", min_value=2.0, max_value=12.0, value=float(CONFIG["media_rentabilidad_bolsa"] * 100), step=0.5)
    CONFIG["media_rentabilidad_bolsa"] = rent_bolsa_pct / 100.0

    inflacion_pct = st.slider("Inflación Media Esperada (%)", min_value=0.5, max_value=6.0, value=float(CONFIG["media_inflacion"] * 100), step=0.25)
    CONFIG["media_inflacion"] = inflacion_pct / 100.0

with st.sidebar.expander("🏛️ Herencia Estocástica", expanded=False):
    CONFIG["herencia_activa"] = st.checkbox("Incluir Herencia en la Simulación", value=CONFIG["herencia_activa"])
    if CONFIG["herencia_activa"]:
        CONFIG["importe_herencia"] = st.number_input("Importe Neto Herencia (€)", value=CONFIG["importe_herencia"], step=25000)
        CONFIG["edad_herencia_media"] = st.number_input("Edad Media de Recepción", value=CONFIG["edad_herencia_media"], step=1)
    else:
        CONFIG["importe_herencia"] = 0

with st.sidebar.expander("🧾 Fiscalidad", expanded=False):
    CONFIG["aplicar_fiscalidad_real"] = st.checkbox("Aplicar IRPF real sobre plusvalías (5 tramos)", value=CONFIG["aplicar_fiscalidad_real"])

with st.sidebar.expander("📄 Guía de Uso", expanded=False):
    ruta_pdf_guia = "Guia_Uso_Simulador_v6.pdf"  # debe estar en la misma carpeta que este script
    try:
        with open(ruta_pdf_guia, "rb") as f:
            st.download_button(
                label="⬇️ Descargar Guía de Uso (PDF)",
                data=f.read(),
                file_name="Guia_Uso_Simulador_v6.pdf",
                mime="application/pdf"
            )
    except FileNotFoundError:
        st.caption("Coloca el archivo 'Guia_Uso_Simulador_v6.pdf' en la misma carpeta que este script para activar la descarga.")

# Nota: st.cache_data necesita que el argumento sea hasheable; convertimos
# la lista de tramos a tupla de tuplas para máxima compatibilidad de caché.
CONFIG["tramos_irpf"] = tuple(CONFIG["tramos_irpf"])

# ==========================================
# 🚀 CÁLCULOS Y RENDERIZADO PRINCIPAL
# ==========================================
umbral_target = CONFIG["umbral_exito_objetivo"]
anos_sin_extra, exito_sin = buscar_anos_necesarios_para_objetivo(CONFIG, umbral_target, con_extras=False)
anos_con_extra, exito_con = buscar_anos_necesarios_para_objetivo(CONFIG, umbral_target, con_extras=True)

st.subheader(f"🎯 Tu Camino a la Libertad Financiera (Objetivo: Tasa de Éxito ≥ {umbral_target:.0f}%)")

col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

with col_kpi1:
    if anos_sin_extra is not None:
        ano_libertad_sin = CONFIG["ano_inicio"] + 1 + anos_sin_extra
        st.metric(
            label="Sin Ingresos Extras",
            value=f"{anos_sin_extra} años más",
            delta=f"Libertad en {ano_libertad_sin} ({exito_sin:.1f}% éxito)",
            delta_color="normal"
        )
    else:
        st.metric(label="Sin Ingresos Extras", value="> 15 años", delta="Requiere más ahorro")

with col_kpi2:
    if anos_con_extra is not None:
        ano_libertad_con = CONFIG["ano_inicio"] + 1 + anos_con_extra
        st.metric(
            label="Con Ingresos Extras",
            value=f"{anos_con_extra} años más",
            delta=f"Libertad en {ano_libertad_con} ({exito_con:.1f}% éxito)",
            delta_color="normal"
        )
    else:
        st.metric(label="Con Ingresos Extras", value="> 15 años", delta="Aumentar ingresos extra")

with col_kpi3:
    if anos_sin_extra is not None and anos_con_extra is not None:
        ahorro_tiempo = anos_sin_extra - anos_con_extra
        if ahorro_tiempo > 0:
            st.metric(label="⚡ Impacto de los Extras", value=f"¡Ahorras {ahorro_tiempo} años!", delta="Adelantas tu jubilación", delta_color="normal")
        elif ahorro_tiempo == 0:
            st.metric(label="⚡ Impacto de los Extras", value="Mismo tiempo", delta="Mayor colchón financiero")
        else:
            st.metric(label="⚡ Impacto de los Extras", value="N/D", delta="-")

st.divider()

resultados, df_res = ejecutar_matriz_escenarios(CONFIG)

escenario_sel_idx = st.selectbox(
    "🎯 Selecciona el escenario activo para auditar sus métricas ejecutivas en detalle:",
    options=range(len(resultados)),
    format_func=lambda i: resultados[i]["nombre"]
)

data_esc = resultados[escenario_sel_idx]

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Tasa de Éxito Global", f"{data_esc['tasa_exito']:.1f}%")
with col_m2:
    st.metric("Mediana Patrimonio Final (P50)", f"{data_esc['p50']:,.0f} €".replace(",", "."))
with col_m3:
    st.metric("Escenario Conservador (P10)", f"{data_esc['p10']:,.0f} €".replace(",", "."))
with col_m4:
    st.metric("Probabilidad Legado (>1M€)", f"{data_esc['prob_1m']:.1f}%")

st.divider()

st.subheader("📊 Tabla Comparativa Global de Escenarios")
st.dataframe(df_res, use_container_width=True)

st.divider()

# 4. GRÁFICO DE ABANICO PROBABILÍSTICO
matriz = data_esc["matriz_trayectorias"]
matriz_infl = data_esc["matriz_factor_inflacion"]
edades = data_esc["edades"]

ver_euros_hoy = st.checkbox("Ajustar por inflación (ver en euros de hoy)", value=False,
                             help="Por defecto el patrimonio se muestra en euros nominales (futuros). Actívalo para ver poder adquisitivo real.")

if ver_euros_hoy:
    matriz_mostrar = matriz / matriz_infl
else:
    matriz_mostrar = matriz

p10_tray = np.percentile(matriz_mostrar, 10, axis=0)
p50_tray = np.percentile(matriz_mostrar, 50, axis=0)
p90_tray = np.percentile(matriz_mostrar, 90, axis=0)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=np.concatenate([edades, edades[::-1]]),
    y=np.concatenate([p90_tray, p10_tray[::-1]]),
    fill='toself',
    fillcolor='rgba(26, 54, 93, 0.15)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    name="Rango Probable (P10 - P90)"
))

fig.add_trace(go.Scatter(
    x=edades, y=p50_tray,
    mode='lines',
    line=dict(color='#1a365d', width=3),
    name="Escenario Central (Mediana P50)"
))

unidad = "euros de hoy" if ver_euros_hoy else "euros nominales"
fig.update_layout(
    title=f"Evolución Probabilística del Patrimonio ({unidad}) - {data_esc['nombre']}",
    xaxis_title="Edad (Años)",
    yaxis_title="Patrimonio Total (€)",
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)
st.caption("Nota metodológica: cada percentil (P10/P50/P90) se calcula de forma independiente para cada edad. La línea P50 no representa el camino de ninguna simulación individual, sino el valor mediano transversal en cada año (técnica estándar de 'fan chart').")

# 5. LÍNEA DEL TIEMPO DE HITOS CLAVE
st.subheader("📅 Línea del Tiempo de Hitos Financieros")

edad_ini = CONFIG["edad_inicial"]
ano_base = CONFIG["ano_inicio"]
anos_t = data_esc["anos_transicion"]

edad_salida = edad_ini + anos_t
ano_salida = ano_base + anos_t

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("🏁 Salida / Retiro", f"{edad_salida} años", f"Año {ano_salida}")
    st.caption("Inicio de etapa de libertad")

with col2:
    if CONFIG["herencia_activa"]:
        st.metric("🏛️ Herencia Estocástica", f"{CONFIG['edad_herencia_media']} años", f"{CONFIG['importe_herencia']:,.0f} €".replace(",", "."))
        st.caption("Entrada estimada de capital")
    else:
        st.metric("🏛️ Herencia Estocástica", "Desactivada", "0 €")

with col3:
    st.metric("💼 Fin Ingresos Extras", f"{CONFIG['edad_fin_ingresos_extras']} años", f"{CONFIG['ingreso_extra_deseado_hoy']:,.0f} €/año".replace(",", "."))
    st.caption("Cierre flujos secundarios")

with col4:
    st.metric("👴 Pensión Pública", f"{CONFIG['edad_retiro_bolsa']} años", f"{CONFIG['pension_base']:,.0f} €/mes".replace(",", "."))
    st.caption("Activación cobro del Estado")

with col5:
    st.metric("🏠 Fin de Hipoteca", f"{CONFIG['edad_fin_hipoteca']} años", f"-{CONFIG['cuota_hipoteca_hoy']:,.0f} €/mes".replace(",", "."))
    st.caption("Reducción de gastos base")

st.divider()

# ==========================================
# 🔬 ANÁLISIS DE SENSIBILIDAD (HEATMAP)
# ==========================================
st.subheader("🔬 Análisis de Sensibilidad")
st.caption(
    "Comprueba qué variables mueven más el resultado del escenario seleccionado arriba "
    f"(**{data_esc['nombre']}**). Se usa la misma semilla aleatoria en todas las celdas "
    "(common random numbers) para que las diferencias reflejen el efecto del parámetro, no ruido estadístico."
)

VARIABLES_SENSIBILIDAD = {
    "Rentabilidad Bolsa (%)": ("media_rentabilidad_bolsa", "pct"),
    "Volatilidad Bolsa (%)": ("volatilidad_bolsa", "pct"),
    "Rentabilidad Renta Fija (%)": ("media_rentabilidad_rf", "pct"),
    "Inflación Media (%)": ("media_inflacion", "pct"),
    "Gasto Mensual (€)": ("gasto_mensual_hoy", "eur"),
}


def _rango_por_defecto(tipo, valor_actual):
    if tipo == "pct":
        centro = valor_actual * 100
        spread = max(3.0, centro * 0.4)
        lo = max(0.1, centro - spread)
        hi = centro + spread
    else:
        centro = valor_actual
        spread = max(200.0, centro * 0.4)
        lo = max(0.0, centro - spread)
        hi = centro + spread
    return float(lo), float(hi)


col_sens1, col_sens2, col_sens3 = st.columns(3)
with col_sens1:
    nombre_var_x = st.selectbox("Variable Eje X", list(VARIABLES_SENSIBILIDAD.keys()), index=0)
with col_sens2:
    opciones_y = [v for v in VARIABLES_SENSIBILIDAD.keys() if v != nombre_var_x]
    nombre_var_y = st.selectbox("Variable Eje Y", opciones_y, index=min(2, len(opciones_y) - 1))
with col_sens3:
    metrica_sel = st.selectbox("Métrica a visualizar", ["Tasa de Éxito (%)", "Patrimonio Mediana P50 (€)"])

col_res1, col_res2 = st.columns(2)
with col_res1:
    resolucion = st.slider("Resolución de la rejilla (puntos por eje)", min_value=3, max_value=11, value=7, step=2)
with col_res2:
    n_sims_heatmap = st.number_input(
        "Simulaciones por celda", min_value=500, max_value=10000, value=3000, step=500,
        help="Menos simulaciones = mapa más rápido pero con más ruido en la Tasa de Éxito. Sube este valor para un informe final."
    )

key_x, tipo_x = VARIABLES_SENSIBILIDAD[nombre_var_x]
key_y, tipo_y = VARIABLES_SENSIBILIDAD[nombre_var_y]

lo_x_def, hi_x_def = _rango_por_defecto(tipo_x, CONFIG[key_x])
lo_y_def, hi_y_def = _rango_por_defecto(tipo_y, CONFIG[key_y])

col_rx1, col_rx2 = st.columns(2)
with col_rx1:
    rango_x = st.slider(
        f"Rango {nombre_var_x}",
        min_value=float(lo_x_def * 0.3), max_value=float(hi_x_def * 1.7),
        value=(lo_x_def, hi_x_def)
    )
with col_rx2:
    rango_y = st.slider(
        f"Rango {nombre_var_y}",
        min_value=float(lo_y_def * 0.3), max_value=float(hi_y_def * 1.7),
        value=(lo_y_def, hi_y_def)
    )

if st.button("🔍 Calcular mapa de sensibilidad"):
    valores_x = np.linspace(rango_x[0], rango_x[1], resolucion)
    valores_y = np.linspace(rango_y[0], rango_y[1], resolucion)

    matriz_z = np.zeros((resolucion, resolucion))
    cfg_base_sens = CONFIG.copy()
    cfg_base_sens["num_simulaciones"] = int(n_sims_heatmap)

    total = resolucion * resolucion
    contador = 0
    barra = st.progress(0, text="Calculando escenarios...")

    for iy, vy in enumerate(valores_y):
        for ix, vx in enumerate(valores_x):
            cfg_celda = cfg_base_sens.copy()
            cfg_celda[key_x] = vx / 100.0 if tipo_x == "pct" else vx
            cfg_celda[key_y] = vy / 100.0 if tipo_y == "pct" else vy
            res_celda = simular_escenario_especifico(
                cfg_celda,
                data_esc["anos_transicion"],
                data_esc["indem"],
                data_esc["ahorro"],
                data_esc["extra"]
            )
            matriz_z[iy, ix] = res_celda["tasa_exito"] if metrica_sel == "Tasa de Éxito (%)" else res_celda["p50"]
            contador += 1
            barra.progress(contador / total, text=f"Calculando escenarios... {contador}/{total}")

    barra.empty()
    st.session_state["heatmap_z"] = matriz_z
    st.session_state["heatmap_x"] = valores_x
    st.session_state["heatmap_y"] = valores_y
    st.session_state["heatmap_labels"] = (nombre_var_x, nombre_var_y, metrica_sel)

if "heatmap_z" in st.session_state:
    matriz_z = st.session_state["heatmap_z"]
    valores_x_plot = st.session_state["heatmap_x"]
    valores_y_plot = st.session_state["heatmap_y"]
    lbl_x, lbl_y, lbl_metrica = st.session_state["heatmap_labels"]

    colorscale = "RdYlGn" if "Éxito" in lbl_metrica else "Viridis"

    fig_heat = go.Figure(data=go.Heatmap(
        z=matriz_z,
        x=[f"{v:.1f}" for v in valores_x_plot],
        y=[f"{v:.1f}" for v in valores_y_plot],
        colorscale=colorscale,
        text=np.round(matriz_z, 1),
        texttemplate="%{text}",
        colorbar=dict(title=lbl_metrica)
    ))
    fig_heat.update_layout(
        title=f"Sensibilidad: {lbl_metrica} según {lbl_x} vs {lbl_y}",
        xaxis_title=lbl_x,
        yaxis_title=lbl_y,
        template="plotly_white"
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption(
        "Cuanto más abrupto sea el cambio de color al moverte por un eje, más sensible es el resultado a esa variable. "
        "Si cambias de escenario o de variables arriba, pulsa de nuevo el botón para recalcular el mapa."
    )