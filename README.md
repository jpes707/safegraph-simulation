# safegraph-simulation

Codebase for SIGSPATIAL ARIC 2020: Foot-Traffic Informed COVID-19 Simulation and Mitigation

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
