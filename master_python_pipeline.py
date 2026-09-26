# ==============================================================================
# Master Thesis Data Pipeline & Empirical Econometrics in Python
# Master's Thesis Project - University of Patras
# Author: Ioannis Andronidis
# ==============================================================================

# 1. Imports & Environment Setup ----------------------------------------------
import warnings
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"


# 2. Auxiliary Functions ------------------------------------------------------
def hausman_test(fe, re):
    """Calculates the Hausman Test statistic between Fixed Effects and Random Effects models."""
    fe_params = fe.params.drop("const", errors="ignore")
    re_params = re.params.drop("const", errors="ignore")

    common_vars = fe_params.index.intersection(re_params.index)
    b_fe = fe_params[common_vars]
    b_re = re_params[common_vars]

    v_fe = fe.cov.loc[common_vars, common_vars]
    v_re = re.cov.loc[common_vars, common_vars]

    b_diff = b_fe - b_re
    v_diff = v_fe - v_re

    chi2 = np.dot(b_diff.T, la.pinv(v_diff).dot(b_diff))
    df_degrees = b_fe.size
    p_value = stats.chi2.sf(chi2, df_degrees)

    return chi2, df_degrees, p_value


def assign_period(year):
    """Categorizes years into macroeconomic and crisis structural periods."""
    if year <= 2019:
        return "Pre-COVID\n(2014-2019)"
    elif 2020 <= year <= 2022:
        return "COVID Period\n(2020-2022)"
    else:
        return "Post-COVID\n(2023-2024)"


# ==============================================================================
# SECTION A: DATA PREPROCESSING, INTERPOLATION & DEFLATION
# ==============================================================================
print(">>> [STAGE 1/4] Starting Data Preprocessing & Deflation...")

# 1. Demand Dataset Imputation
df_demand = pd.read_excel("demand_factors_data.xlsx", sheet_name="Data Sheet")

df_demand["Touremp"] = pd.to_numeric(
    df_demand["Touremp"].astype(str).str.replace(",", "."), errors="coerce"
)
df_demand["Accom"] = pd.to_numeric(
    df_demand["Accom"].astype(str).str.replace(",", "."), errors="coerce"
)

df_demand["Touremp"] = df_demand["Touremp"].interpolate(method="linear")
df_demand["Accom"] = df_demand["Accom"].interpolate(method="linear")
df_demand["Tourist_Arrivals"] = df_demand["Tourist_Arrivals"].interpolate(
    method="linear"
)

df_demand.to_excel("demand_data_complete.xlsx", index=False)

# 2. Efficiency Dataset Imputation & HICP Adjustment
df_sfa = pd.read_excel("SFA_Data.xlsx", sheet_name="Data Sheet")

for col in [
    "Passengers",
    "Cargo_Tonnes",
    "WLU(Output)",
    "Movements",
    "labor_Cost",
    "Labor Number",
]:
    df_sfa[col] = df_sfa[col].interpolate(method="linear")

df_sfa.to_excel("efficiency_data_complete.xlsx", index=False)

df_eff = pd.read_excel("efficiency_data_complete.xlsx")
df_eff["HICP"] = df_eff["HICP"].interpolate(method="linear")
df_eff.to_excel("efficiency_thesis_data.xlsx", index=False)

# 3. Country Mapping & Deflation via HICP
airport_to_country = {
    "BRUSSELS airport": "Belgium",
    "ANTWERPEN/DEURNE airport": "Belgium",
    "FRANKFURT/MAIN airport": "Germany",
    "MUNICH airport": "Germany",
    "DUBLIN airport": "Ireland",
    "ATHINAI/ELEFTHERIOS VENIZELOS airport": "Greece",
    "IOANNINA/KING PYRROS airport": "Greece",
    "KOS/IPPOKRATIS airport": "Greece",
    "KERKIRA/IOANNIS KAPODISTRIAS airport": "Greece",
    "MIKONOS airport": "Greece",
    "NAXOS airport": "Greece",
    "SKIATHOS/ALEXANDROS PAPADIAMANDIS airport": "Greece",
    "SANTORINI airport": "Greece",
    "THESSALONIKI/MAKEDONIA airport": "Greece",
    "BARCELONA/EL PRAT airport": "Spain",
    "ADOLFO SUAREZ MADRID-BARAJAS airport": "Spain",
    "LA PALMA airport": "Spain",
    "LANZAROTE airport": "Spain",
    "TENERIFE NORTE airport": "Spain",
    "BILBAO airport": "Spain",
    "IBIZA airport": "Spain",
    "MALAGA/COSTA DEL SOL airport": "Spain",
    "SAN SEBASTIAN airport": "Spain",
    "SEVILLA airport": "Spain",
    "PARIS-CHARLES DE GAULLE airport": "France",
    "PARIS-ORLY airport": "France",
    "DZAOUDZI airport": "France",
    "CALVI-SAINTE-CATHERINE airport": "France",
    "CHAMBERY-AIX-LES-BAINS airport": "France",
    "MARSEILLE-PROVENCE airport": "France",
    "LILLE-LESQUIN airport": "France",
    "ROMA/FIUMICINO airport": "Italy",
    "COMISO airport": "Italy",
    "ALGHERO/FERTILIA airport": "Italy",
    "CAGLIARI/ELMAS airport": "Italy",
    "MILANO/MALPENSA airport": "Italy",
    "AMSTERDAM/SCHIPHOL airport": "Netherlands",
    "SOFIA airport": "Bulgaria",
    "VARNA airport": "Bulgaria",
    "IVALO airport": "Finland",
    "JOENSUU airport": "Finland",
    "JYVASKYLA airport": "Finland",
    "KEFLAVIK airport": "Iceland",
    "BARDUFOSS airport": "Norway",
    "HARSTAD/NARVIK/EVENES airport": "Norway",
    "HAUGESUND/KARMOY airport": "Norway",
    "KIRKENES/HOYBUKTMOEN airport": "Norway",
    "GENEVA airport": "Switzerland",
    "SION airport": "Switzerland",
    "ZURICH airport": "Switzerland",
    "BALE-MULHOUSE airport": "Switzerland",
}

df_eff["Country"] = df_eff["Airport"].map(airport_to_country)

# Vectorized Deflation
df_eff["Labor Cost"] = (df_eff["lC_OLD"] / df_eff["HICP"]) * 100
df_eff["Operating Cost"] = (df_eff["Op Cost old"] / df_eff["HICP"]) * 100
df_eff["Aeronautical Revenues"] = (df_eff["Aero Revenues"] / df_eff["HICP"]) * 100
df_eff["Non-Aeronautical Revenues"] = (
    df_eff["Non-AeroRevenues"] / df_eff["HICP"]
) * 100
df_eff["Total Revenues"] = (df_eff["Total Rev"] / df_eff["HICP"]) * 100

df_eff.to_excel("eff_data_final.xlsx", index=False)
print(">>> [STAGE 1/4] Preprocessing & Deflation Completed Successfully.\n")


# ==============================================================================
# SECTION B: AIR PASSENGER TRANSPORT DEMAND ECONOMETRIC ANALYSIS
# ==============================================================================
print(">>> [STAGE 2/4] Executing Econometric Panel Regressions...")

df_demand_proc = pd.read_excel("demand_data_complete.xlsx")

numeric_cols = [
    "passengers",
    "education",
    "real_gdp_per_capita",
    "urban_population",
    "Total population",
    "Tourist_Arrivals",
    "Accom",
    "Touremp",
    "CPI",
    "Exchange rate",
    "Jet Fuel Price",
    "Covid_19",
    "PPP",
]

# String cleaning & Numeric formatting
for col in numeric_cols:
    if col in df_demand_proc.columns:
        df_demand_proc[col] = (
            df_demand_proc[col]
            .astype(str)
            .str.replace(r"\xa0", "", regex=True)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df_demand_proc[col] = pd.to_numeric(
            df_demand_proc[col], errors="coerce"
        )

# Heatmap Plot
plt.figure(figsize=(11, 9))
cor_matrix = df_demand_proc[numeric_cols].corr()
sns.heatmap(cor_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Macroeconomic Demand Drivers", fontweight="bold")
plt.tight_layout()
plt.show()

# Panel Data Structure & Log Transformations
df_panel = df_demand_proc.set_index(["country_name", "year"])
df_panel = df_panel.rename(columns={"Jet Fuel Price": "tones of jet fuel"})

df_panel["log_passengers"] = np.log(df_panel["passengers"])
df_panel["log_pop"] = np.log(df_panel["Total population"])
df_panel["log_gdp"] = np.log(df_panel["real_gdp_per_capita"])
df_panel["log_tourist"] = np.log(df_panel["Tourist_Arrivals"])
df_panel["log_accom"] = np.log(df_panel["Accom"])
df_panel["log_urb"] = np.log(df_panel["urban_population"])

# 1. Random Effects Models
print("\n" + "=" * 60)
print("--- MODEL 1: Random Effects (Population Specification) ---")
ind_var1_log = [
    "log_pop",
    "log_gdp",
    "tones of jet fuel",
    "CPI",
    "education",
    "Exchange rate",
    "Covid_19",
    "geo",
    "log_urb",
    "PPP",
]
X_log1 = sm.add_constant(df_panel[ind_var1_log])
model_re1 = RandomEffects(df_panel["log_passengers"], X_log1)
res_re1 = model_re1.fit(cov_type="clustered", cluster_entity=True)
print(res_re1.summary)

print("\n" + "=" * 60)
print("--- MODEL 2: Random Effects (Tourist Arrivals Specification) ---")
ind_var2_log = [
    "education",
    "log_gdp",
    "log_urb",
    "log_tourist",
    "CPI",
    "Exchange rate",
    "tones of jet fuel",
    "Covid_19",
    "PPP",
    "geo",
]
X_log2 = sm.add_constant(df_panel[ind_var2_log])
model_re2 = RandomEffects(df_panel["log_passengers"], X_log2)
res_re2 = model_re2.fit(cov_type="clustered", cluster_entity=True)
print(res_re2.summary)

print("\n" + "=" * 60)
print("--- MODEL 3: Random Effects (Accommodations Specification) ---")
ind_var3_log = [
    "education",
    "log_gdp",
    "log_urb",
    "CPI",
    "Exchange rate",
    "tones of jet fuel",
    "geo",
    "Covid_19",
    "PPP",
    "log_accom",
]
X_log3 = sm.add_constant(df_panel[ind_var3_log])
model_re3 = RandomEffects(df_panel["log_passengers"], X_log3)
res_re3 = model_re3.fit(cov_type="clustered", cluster_entity=True)
print(res_re3.summary)

# Interaction Term Regression
print("\n" + "=" * 60)
print("--- MODEL 4: Random Effects with Interaction (Tourism * Geography) ---")
df_panel["Tour_X_Geo"] = df_panel["log_tourist"] * df_panel["geo"]
X_vars_inter = df_panel[
    [
        "education",
        "log_gdp",
        "log_urb",
        "log_tourist",
        "CPI",
        "Exchange rate",
        "tones of jet fuel",
        "Covid_19",
        "PPP",
        "Tour_X_Geo",
    ]
]
model_inter = RandomEffects(df_panel["log_passengers"], X_vars_inter)
res_inter = model_inter.fit(cov_type="clustered", cluster_entity=True)
print(res_inter.summary)

# 2. Fixed Effects Models
print("\n" + "=" * 60)
print("--- FIXED EFFECTS ESTIMATIONS ---")

ind_var1_fe = [
    "log_pop",
    "log_gdp",
    "tones of jet fuel",
    "CPI",
    "education",
    "Exchange rate",
    "Covid_19",
    "log_urb",
    "PPP",
]
model_fe1 = PanelOLS(
    df_panel["log_passengers"],
    sm.add_constant(df_panel[ind_var1_fe]),
    entity_effects=True,
)
res_fe1 = model_fe1.fit(cov_type="clustered", cluster_entity=True)

ind_var2_fe = [
    "education",
    "log_gdp",
    "log_urb",
    "log_tourist",
    "CPI",
    "Exchange rate",
    "tones of jet fuel",
    "Covid_19",
    "PPP",
]
model_fe2 = PanelOLS(
    df_panel["log_passengers"],
    sm.add_constant(df_panel[ind_var2_fe]),
    entity_effects=True,
)
res_fe2 = model_fe2.fit(cov_type="clustered", cluster_entity=True)

ind_var3_fe = [
    "education",
    "log_gdp",
    "log_urb",
    "CPI",
    "Exchange rate",
    "tones of jet fuel",
    "Covid_19",
    "PPP",
    "log_accom",
]
model_fe3 = PanelOLS(
    df_panel["log_passengers"],
    sm.add_constant(df_panel[ind_var3_fe]),
    entity_effects=True,
)
res_fe3 = model_fe3.fit(cov_type="clustered", cluster_entity=True)

# Hausman Test Example (FE vs RE for Model 2)
chi2_stat, df_deg, pval = hausman_test(res_fe2, res_re2)
print(
    f"\nHausman Test (Model 2) -> Chi2: {chi2_stat:.4f}, df: {df_deg}, p-value: {pval:.4f}"
)


# ==============================================================================
# SECTION C: EXPLORATORY VISUALIZATIONS & DISCRIPTIVE BOXPLOTS
# ==============================================================================
print("\n>>> [STAGE 3/4] Generating Macroeconomic Dynamics Plots...")

# 1. Passengers Lineplot per Country
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_demand_proc,
    x="year",
    y="passengers",
    hue="country_name",
    marker="o",
    linewidth=2,
)
plt.title(
    "Air Passenger Traffic Trends Across European Countries",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Year", fontsize=11)
plt.ylabel("Passengers", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Countries")
plt.tight_layout()
plt.show()

# 2. Comparative 3x3 Boxplots Panel (Pre-COVID, COVID, Post-COVID)
df_tobit = pd.read_csv("tobit_data.csv")
df_tobit.columns = df_tobit.columns.str.strip()

df_tobit["Period"] = df_tobit["Year"].apply(assign_period)
period_order = [
    "Pre-COVID\n(2014-2019)",
    "COVID Period\n(2020-2022)",
    "Post-COVID\n(2023-2024)",
]

# Rescaling variables
df_tobit["Tourist_Arrivals_M"] = df_tobit["Tourist_Arrivals"] / 1e6
df_tobit["Total_population_M"] = df_tobit["Total_population"] / 1e6
df_tobit["Accom_K"] = df_tobit["Accom"] / 1e3

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

variables_config = [
    (
        "real_gdp_per_capita",
        "(a) Real GDP per Capita (€)",
        "€",
        "Greens",
        axes[0, 0],
    ),
    (
        "Tourist_Arrivals_M",
        "(b) Tourist Arrivals (M)",
        "Millions",
        "Reds",
        axes[0, 1],
    ),
    (
        "Total_population_M",
        "(c) Total Population (M)",
        "Millions",
        "Purples",
        axes[0, 2],
    ),
    (
        "urban_population",
        "(d) Urban Population (%)",
        "%",
        "Blues",
        axes[1, 0],
    ),
    ("education", "(e) Education Level (%)", "%", "YlGn", axes[1, 1]),
    ("CPI", "(f) Consumer Price Index (CPI)", "Rate (%)", "Oranges", axes[1, 2]),
    ("Exchange rate", "(g) Exchange Rate", "Index", "Greys", axes[2, 0]),
    ("PPP", "(h) Purchasing Power Parity", "Ratio", "GnBu", axes[2, 1]),
    (
        "Accom_K",
        "(i) Accommodation Capacity (K)",
        "Thousands",
        "PuBu",
        axes[2, 2],
    ),
]

for var, title, ylabel, palette, ax in variables_config:
    sns.boxplot(
        ax=ax,
        x="Period",
        y=var,
        hue="Period",
        data=df_tobit,
        order=period_order,
        palette=palette,
        legend=False,
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)

plt.tight_layout()
plt.savefig("tobit_macro_variables_panel.png", dpi=300, bbox_inches="tight")
plt.show()

print(
    ">>> [STAGE 4/4] Pipeline Completed Successfully! Plots and files exported."
)

