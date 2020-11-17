# safegraph-simulation

Codebase for SIGSPATIAL ARIC 2020: Data-Driven Mobility Models for COVID-19 Simulation (https://doi.org/10.1145/3423455.3430305/)

## Abstract

Agent-based models (ABM) play a prominent role in guiding critical decision-making and supporting the development of effective policies for better urban resilience and response to the COVID-19 pandemic. However, many ABMs lack realistic representations of human mobility, a key process that leads to physical interaction and subsequent spread of disease. Therefore, we propose the application of Latent Dirichlet Allocation (LDA), a topic modeling technique, to foot-traffic data to develop a realistic model of human mobility in an ABM that simulates the spread of COVID-19. In our novel approach, LDA treats POIs as “words” and agent home census block groups (CBGs) as “documents” to extract “topics” of POIs that frequently appear together in CBG visits. These topics allow us to simulate agent mobility based on the LDA topic distribution of their home CBG. We compare the LDA based mobility model with competitor approaches including a naive mobility model that assumes visits to POIs are random. We find that the naive mobility model is unable to facilitate the spread of COVID-19 at all. Using the LDA informed mobility model, we simulate the spread of COVID-19 and test the effect of changes to the number of topics, various parameters, and public health interventions. By examining the simulated number of cases over time, we find that the number of topics does indeed impact disease spread dynamics, but only in terms of the outbreak’s timing. Further analysis of simulation results is needed to better understand the impact of topics on simulated COVID-19 spread. This study contributes to strengthening human mobility representations in ABMs of disease spread.

## Installation

1. Clone the repository.

2. `pip install -r requirements.txt`.

3. Create a subdirectory called `safegraph-data` in the repository directory.

4. Download the SafeGraph Open Census Data [here](https://www.safegraph.com/open-census-data). Save it to `safegraph-data/safegraph_open_census_data`.

5. Get free access to the remaining SafeGraph data by following [these steps](https://www.safegraph.com/covid-19-data-consortium). A non-disclosure agreement must be signed as part of the process.

6. Download the SafeGraph Weekly Patterns (v2) data. Save it to `safegraph-data/safegraph_weekly_patterns_v2`.

7. Download the SafeGraph Core Places data. Save it to `safegraph-data/safegraph_core_places`.

8. Download the SafeGraph Social Distancing Metrics data. Save it to `safegraph-data/safegraph_social_distancing_metrics`.

## Usage

* `county-parser.py` is used to extract locality data. Run this at least once, and then run `simulation.py`. To create custom simulation configurations, place new `.cfg` files in `config-files/`. Use `config-files/default-config.cfg` as a template.

* `simulation-naive.py` is used to simulate community and household spread in a naive way by weighting POIs equally, ignoring the time of day, and using a constant dwell time. In most cases, the virus will only reach a very small number of agents because agents will not congregate at popular POIs, making this simulation version unrealistic.

* `mobility-stats.py` is used to evaluate mobility data assuming no virus has been introduced. It provides the total number of visitors to each POI within a specified timeframe.

* `sigspatial-trials.txt`, `sigspatial-trial-runner.py`, and any files in a folder that is titled `sigspatial-trials/` were used to generate results for the conference publication. These files will likely not be applicable in other cases.

* All results will be located in `results/`.

## Links

* DOI: https://dl.acm.org/doi/proceedings/10.1145/3356395/
* Paper: https://urbands.github.io/aric2020/ARIC2020_Paper4.pdf/
* Interactive visualizations: https://jpes707.github.io/safegraph-simulation/
