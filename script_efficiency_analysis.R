# ==============================================================================
# Master's Thesis: European Airport Efficiency & Demand Analysis
# Methodology: DEA (Stage 1), Malmquist Index & Panel Tobit Regression (Stage 2)
# Author: Ioannis Andronidis
# ==============================================================================

# 1. Environment Setup & Libraries --------------------------------------------
setwd("/Users/user/Desktop/thesis/datasets/sfa")

library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(GGally)
library(corrplot)
library(Benchmarking)
library(plm)
library(censReg)
library(lmtest)
library(sandwich)

options(scipen = 999)

# 2. Data Importing & Cleaning -------------------------------------------------
raw_data <- read_excel("eff_data_final.xlsx")

data <- raw_data %>%
  as.data.frame() %>%
  select(-HICP, -Years, -Country) %>%
  rename(
    terminal                 = `Terminals(input 1)`,
    runways                  = `Runways(input 2)`,
    gates                    = `Gates(input 3)`,
    stands                   = `Aircraft Stands(input 4)`,
    Aeronautical_revenues   = `Aeronautical Revenues`,
    Non_aeronautial_revenues = `Non-Aeronautical Revenues`,
    Labor_Number             = `Labor Number`,
    Total_Revenues           = `Total Revenues`,
    Operating_Cost           = `Operating Cost`
  )

# 3. Correlation Analysis ------------------------------------------------------
vars_for_corr <- c(
  "Cargo_Tonnes", "Movements", "runways", "stands", "Operating_Cost",
  "Non_aeronautial_revenues", "Total_Revenues", "Passengers", "WLU(Output)",
  "terminal", "gates", "labor_Cost", "Aeronautical_revenues", "Labor_Number"
)

# Ensure numeric format across feature subset
for (col in vars_for_corr) {
  data[, col] <- as.numeric(as.character(data[, col]))
}

data_subset <- data[, vars_for_corr]
cor_matrix  <- cor(data_subset, use = "pairwise.complete.obs")

# Correlation matrix visual
corrplot(
  cor_matrix,
  method       = "square",
  type         = "full",
  order        = "original",
  title        = "Correlation Heat Map of Inputs and Outputs",
  addCoef.col  = "black",
  number.cex   = 0.5,
  mar          = c(0, 0, 2, 0),
  tl.col       = "black",
  tl.cex       = 0.6,
  tl.srt       = 45
)

# Variable selection based on multicollinearity reduction:
# Outputs: WLU(Output), Movements
# Inputs: labor_Cost, Operating_Cost, runways, stands, gates

data_dea <- data %>%
  select(-Labor_Number, -Passengers, -Cargo_Tonnes, -Total_Revenues, 
         -Aeronautical_revenues, -Non_aeronautial_revenues)

# 4. Data Envelopment Analysis (DEA) -------------------------------------------
inputs_variables  <- data_dea[, c("labor_Cost", "Operating_Cost", "runways", "gates", "stands")]
outputs_variables <- data_dea[, c("WLU(Output)", "Movements")]

# Output-oriented DEA Models (VRS and CRS)
dea_model_vrs <- dea(inputs_variables, outputs_variables, RTS = "vrs", ORIENTATION = "out")
dea_model_crs <- dea(inputs_variables, outputs_variables, RTS = "crs", ORIENTATION = "out")

# Efficiency scores evaluation
data_dea$Efficiency_Score_Vrs <- 1 / eff(dea_model_vrs)
data_dea$Efficiency_score_crs <- 1 / eff(dea_model_crs)
data_dea$Scale_Efficiency     <- data_dea$Efficiency_score_crs / data_dea$Efficiency_Score_Vrs

# Slacks computation
slacks_vrs <- slack(inputs_variables, outputs_variables, dea_model_vrs)
slacks_crs <- slack(inputs_variables, outputs_variables, dea_model_crs)

print(slacks_vrs)
print(slacks_crs)

# 5. DEA Visualizations & Frontier Trends --------------------------------------

# 5.1 Mean Efficiency Trend (2014-2024)
mean_trends <- data_dea %>%
  mutate(Year = as.numeric(as.character(Year))) %>%
  group_by(Year) %>%
  summarise(
    TE_CRS           = mean(Efficiency_score_crs, na.rm = TRUE),
    PTE_VRS          = mean(Efficiency_Score_Vrs, na.rm = TRUE),
    Scale_Efficiency = mean(Scale_Efficiency, na.rm = TRUE)
  ) %>%
  pivot_longer(
    cols      = c(TE_CRS, PTE_VRS, Scale_Efficiency),
    names_to  = "Efficiency_Type",
    values_to = "Mean_Score"
  )

ggplot(mean_trends, aes(x = Year, y = Mean_Score, color = Efficiency_Type)) +
  geom_line(aes(group = Efficiency_Type), size = 1.2) +
  geom_point(size = 2.5) +
  scale_x_continuous(breaks = seq(2014, 2024, by = 1)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.1)) +
  scale_color_manual(
    values = c("TE_CRS" = "#1f77b4", "PTE_VRS" = "#ff7f0e", "Scale_Efficiency" = "#2ca02c"),
    labels = c("TE_CRS" = "Technical Efficiency (CRS)", 
               "PTE_VRS" = "Pure Technical Efficiency (VRS)", 
               "Scale_Efficiency" = "Scale Efficiency (SE)")
  ) +
  theme_minimal() +
  theme(
    plot.title     = element_text(face = "bold", size = 14, hjust = 0.5),
    axis.title     = element_text(face = "bold", size = 11),
    legend.position = "bottom",
    legend.title    = element_blank(),
    panel.grid.minor = element_blank()
  ) +
  labs(
    title = "Mean Efficiency Trends of European Airports (2014-2024)",
    x     = "Year",
    y     = "Mean Efficiency Score"
  )

# 5.2 Efficiency Score Distribution Boxplot
ggplot(data_dea, aes(x = as.factor(Year), y = Efficiency_Score_Vrs)) +
  geom_boxplot(fill = "#f8f9fa", color = "#2c3e50", outlier.shape = NA, alpha = 0.6) +
  geom_jitter(color = "#ff7f0e", width = 0.2, alpha = 0.4, size = 1.5) +
  theme_minimal() +
  labs(
    title    = "Distribution of Pure Technical Efficiency (VRS) Per Year",
    subtitle = "Analysis of 51 European Airports over 2014-2024",
    x        = "Year",
    y        = "VRS Efficiency Score"
  ) +
  theme(
    plot.title    = element_text(face = "bold", size = 13, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    axis.title    = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

# 5.3 DEA Frontier Visuals: COVID Shock vs Recovery
data_2020 <- subset(data_dea, Year == 2020)
data_2024 <- subset(data_dea, Year == 2024)

par(mfrow = c(1, 2))

# 2020 Frontier
dea.plot(data_2020$Operating_Cost, data_2020$`WLU(Output)`, 
         RTS = "vrs", ORIENTATION = "out", txt = TRUE, 
         xlab = "Operating Cost", ylab = "Work Load Units (WLU)",
         main = "DEA Frontier - COVID Shock (2020)", col = "red", lwd = 2)
dea.plot(data_2020$Operating_Cost, data_2020$`WLU(Output)`, 
         RTS = "crs", ORIENTATION = "out", add = TRUE, lty = "dashed", col = "darkred")

# 2024 Frontier
dea.plot(data_2024$Operating_Cost, data_2024$`WLU(Output)`, 
         RTS = "vrs", ORIENTATION = "out", txt = TRUE, 
         xlab = "Operating Cost", ylab = "Work Load Units (WLU)",
         main = "DEA Frontier - Full Recovery (2024)", col = "blue", lwd = 2)
dea.plot(data_2024$Operating_Cost, data_2024$`WLU(Output)`, 
         RTS = "crs", ORIENTATION = "out", add = TRUE, lty = "dashed", col = "darkblue")

par(mfrow = c(1, 1))

# 6. Malmquist Productivity Index (MPI) ---------------------------------------
data_sorted <- data_dea %>% arrange(Airport, Year)

X_mpi <- as.matrix(data_sorted[, c("labor_Cost", "Operating_Cost", "runways", "gates", "stands")])
Y_mpi <- as.matrix(data_sorted[, c("WLU(Output)", "Movements")])

mpi_calc <- malmquist(
  X_mpi, Y_mpi, 
  ID   = data_sorted$Airport, 
  TIME = data_sorted$Year, 
  RTS  = "crs"
)

mpi_results <- data.frame(
  Airport           = mpi_calc$id,
  Year_To           = mpi_calc$time,
  Malmquist_Index   = mpi_calc$m,
  Technical_Change  = mpi_calc$tc,
  Efficiency_Change = mpi_calc$ec
)

mpi_summary_table <- mpi_results %>%
  filter(!is.na(Malmquist_Index)) %>%
  group_by(Year_To) %>%
  summarise(
    Mean_Malmquist_Index   = mean(Malmquist_Index, na.rm = TRUE),
    Mean_Technical_Change  = mean(Technical_Change, na.rm = TRUE),
    Mean_Efficiency_Change = mean(Efficiency_Change, na.rm = TRUE)
  ) %>%
  mutate(Period = paste0(Year_To - 1, "-", Year_To)) %>%
  select(Period, Mean_Malmquist_Index, Mean_Technical_Change, Mean_Efficiency_Change)

print(mpi_summary_table)

# MPI Trends Plot
mpi_plot_data <- mpi_summary_table %>%
  pivot_longer(
    cols      = starts_with("Mean_"), 
    names_to  = "Index_Type", 
    values_to = "Value"
  )

ggplot(mpi_plot_data, aes(x = Period, y = Value, group = Index_Type, color = Index_Type)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  geom_hline(yintercept = 1.0, linetype = "dashed", color = "gray50") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_color_manual(
    values = c("Mean_Malmquist_Index" = "#1f77b4", 
               "Mean_Technical_Change" = "#ff7f0e", 
               "Mean_Efficiency_Change" = "#2ca02c"),
    labels = c("Total Productivity (MPI)", "Catch-up (Efficiency Change)", "Frontier Shift (Technical Change)")
  ) +
  labs(
    title    = "Evolution of Malmquist Productivity Indices (2014-2024)",
    subtitle = "Values above 1.0 indicate improvement, below 1.0 indicate decline",
    x        = "Time Period",
    y        = "Index Value",
    color    = "Components"
  )

write.csv(data_dea, file = "efficiency_data_completed_for_analysis.csv", row.names = FALSE)

# 7. Second Stage: Panel Tobit Regression Setup --------------------------------
data_macro_vars <- read_excel("demand_data_complete.xlsx")

airport_country_dict <- data.frame(
  Airport = c(
    "ADOLFO SUAREZ MADRID-BARAJAS airport", "ALGHERO/FERTILIA airport",
    "AMSTERDAM/SCHIPHOL airport", "ANTWERPEN/DEURNE airport",
    "ATHINAI/ELEFTHERIOS VENIZELOS airport", "BALE-MULHOUSE airport",
    "BARCELONA/EL PRAT airport", "BARDUFOSS airport",
    "BILBAO airport", "BRUSSELS airport",
    "CAGLIARI/ELMAS airport", "CALVI-SAINTE-CATHERINE airport",
    "CHAMBERY-AIX-LES-BAINS airport", "COMISO airport",
    "DUBLIN airport", "DZAOUDZI airport",
    "FRANKFURT/MAIN airport", "GENEVA airport",
    "HARSTAD/NARVIK/EVENES airport", "HAUGESUND/KARMOY airport",
    "IBIZA airport", "IOANNINA/KING PYRROS airport",
    "IVALO airport", "JOENSUU airport",
    "JYVASKYLA airport", "KEFLAVIK airport",
    "KERKIRA/IOANNIS KAPODISTRIAS airport", "KIRKENES/HOYBUKTMOEN airport",
    "KOS/IPPOKRATIS airport", "LA PALMA airport",
    "LANZAROTE airport", "LILLE-LESQUIN airport",
    "MALAGA/COSTA DEL SOL airport", "MARSEILLE-PROVENCE airport",
    "MIKONOS airport", "MILANO/MALPENSA airport",
    "MUNICH airport", "NAXOS airport",
    "PARIS-CHARLES DE GAULLE airport", "PARIS-ORLY airport",
    "ROMA/FIUMICINO airport", "SAN SEBASTIAN airport",
    "SANTORINI airport", "SEVILLA airport",
    "SION airport", "SKIATHOS/ALEXANDROS PAPADIAMANDIS airport",
    "SOFIA airport", "TENERIFE NORTE airport",
    "THESSALONIKI/MAKEDONIA airport", "VARNA airport",
    "ZURICH airport"
  ),
  country_name = c(
    "Spain", "Italy", "Netherlands", "Belgium", "Greece", "France",
    "Spain", "Norway", "Spain", "Belgium", "Italy", "France",
    "France", "Italy", "Ireland", "France", "Germany", "Switzerland",
    "Norway", "Norway", "Spain", "Greece", "Finland", "Finland",
    "Finland", "Iceland", "Greece", "Norway", "Greece", "Spain",
    "Spain", "France", "Spain", "France", "Greece", "Italy",
    "Germany", "Greece", "France", "France", "Italy", "Spain",
    "Greece", "Spain", "Switzerland", "Greece", "Bulgaria", "Spain",
    "Greece", "Bulgaria", "Switzerland"
  ),
  stringsAsFactors = FALSE
)

# Dataset merging
final_dataset <- data_dea %>%
  left_join(airport_country_dict, by = "Airport") %>%
  left_join(data_macro_vars, by = c("country_name" = "country_name", "Year" = "year"))

# Filtering & Variable Standardization
final_dataset_clean <- final_dataset %>%
  filter(Year >= 2018) %>%
  rename(Total_population = `Total population`)

write.csv(final_dataset_clean, "tobit_data.csv", row.names = FALSE)

# 8. Panel Tobit Estimation ----------------------------------------------------
pdata <- pdata.frame(final_dataset_clean, index = c("Airport", "Year"))

model_formula <- Efficiency_Score_Vrs ~ log(real_gdp_per_capita) + 
  log(Tourist_Arrivals) + 
  log(Accom) + 
  log(Total_population) + 
  Covid_19 + 
  geo + 
  education

tobit_re <- censReg(model_formula, right = 1, data = pdata, method = "BHHH")
summary(tobit_re)

