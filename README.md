# Air Transport Efficiency and Demand Factors

##  Overview
This project presents a two-stage empirical econometric study analyzing the determinants of air passenger transport demand and operational efficiency across **51 European airports** over the period **2014–2024**. 

The goal is to evaluate technical efficiency scores and examine the impact of macroeconomic and operational factors on European airport infrastructure.

##  Tech Stack & Methodology
* **Programming Languages:** R, Python
* **Key R Packages:** `Benchmarking`, `censReg`, `plm`, `tidyverse`
* **Methodology:**
  * **Stage 1:** Data Envelopment Analysis (DEA) to derive technical efficiency scores.
  * **Stage 2:** Random Effects Panel Tobit Regression to evaluate operational and macroeconomic drivers.
  * **Documentation:** Compiled in LaTeX / Overleaf.

##  Dataset
The analysis is based on panel data compiled from **Eurostat** and official airport financial/operational reports, covering:
* **Inputs:** Terminal area, runway length, number of gates/check-in desks.
* **Outputs:** Annual passenger volume, cargo traffic, total aircraft movements.
* **Environmental Variables:** Regional GDP, population density.

##  Key Findings
This study evaluates the operational efficiency and passenger demand determinants across 51 European airports from 2014 to 2024 using a two-stage empirical framework. In the first stage, Data Envelopment Analysis (DEA) is applied to calculate technical efficiency scores under both Constant Returns to Scale (CRS) and Variable Returns to Scale (VRS), revealing substantial potential for capacity optimization and highlighting that large international hubs consistently outperform regional facilities. In the second stage, a Random Effects Panel Tobit regression model identifies key macroeconomic and operational drivers, demonstrating that GDP per capita, regional population density, and rail connectivity exert a statistically significant positive effect on airport efficiency scores.
##  Repository Structure
##  Author
**Ioannis Andronidis**
* MSc in Applied Economics & Data Analysis | University of Patras
* BSc in Mathematics | University of Patras
