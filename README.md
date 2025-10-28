# WorkZoneWatch: Predict Traffic Delay Impact from Roadwork on I-10 in Phoenix, AZ

Team members: [Yanbing Wang](https://github.com/yanb514)


## Table of Content
- [Introduction](#introduction)
- [Dataset Generation](#dataset-generation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Future Work](#future-work)
- [Quick Start](#quick-start)
- [Description of Repository](#description-of-repository)

### Introduction
Phoenix’s Interstate 10 Broadway Curve is one of Arizona’s busiest freeway corridors, and its 11-mile segment recently underwent a four-year [reconstruction project](https://azdot.gov/i-10-broadway-curve-project) involving major lane closures between Loop 202 and I-17. *WorkZoneWatch* aims to quantify and predict the short-term delay impacts of planned roadwork and unplanned incidents along this corridor to support ADOT’s construction scheduling and traffic management decisions. Using INRIX historical speed data and AZ511 work zone and incident reports, we built an augmented dataset incorporating road geometry, traffic periodicity, and lagged travel-time features, and tested a suite of models to predict segment-level travel times. A counterfactual analysis was then performed by removing event-influenced data and event-related features to estimate “no-event” conditions, which serve as a baseline for quantifying delay caused by roadwork. Results show that planned events induce measurable but lagged delays—up to 15 seconds per mile (~3 minutes across the 11-mile segment)—roughly double routine congestion levels. Delays are most disruptive during weekday early afternoons and weekend mornings. These findings highlight opportunities to optimize work-zone scheduling and underscore the need for more complete and timely event reporting.

![i10](images/map-broadway-curve-051122.jpg)
*Study Area: I-10 Broadway Curve: 11-mile stretch between Loop 202 and I-17 (source: ADOT)*

## Dataset Generation

Our dataset integrates **AZ511 event reports** and **INRIX historical traffic speed data** to quantify the delay impact of roadwork on the I-10 Broadway Curve corridor. These two datasets are combined through spatial and temporal matching to produce a unified, segment-level time series of traffic and event activity.

- [AZ511 Events Report](https://az511.com/api/wzdx):The AZ511 Traveler Information Platform provides event-based data on scheduled work zones and unplanned incidents across Arizona through the WZDx API. Since June 2025, we have developed and maintained an automated data pipeline that collects 48,109 event records every three hours via API requests and stores them in a local SQLite database for ongoing analysis. Each record includes attributes such as location, direction, timing, and severity following the WZDx specification. To integrate this information with traffic data, we performed geolocation by mapping 474 events on I-10 using their latitude–longitude coordinates to the nearest INRIX Traffic Messaging Center (TMC) segments in both travel directions, creating a spatially aligned event dataset for the I-10 Broadway Curve corridor.

- [INRIX Historic Traffic Speed Data](https://inrix.com/products/ai-traffic/): INRIX provides minute-level observations of vehicle speed, free-flow speed, and travel time for individual TMC segments derived from floating-car GPS data. The dataset spans September 2024–October 2025 and covers 50 TMC segments along I-10. Missing observations were interpolated using temporal smoothing, and duplicate segments were removed. The time range was aligned with AZ511 event records (June–October 2025). Both event and speed data were then synchronized and aggregated into uniform one-hour intervals. This data is accessed through a paid license.

The final dataset forms a 3D tensor with dimensions (50 TMC segments × 2,374 time bins × 20 features). Augmented features include road geometry (segment length, ramps, curvature), cyclic time variables (hour of day and day of week), lagged travel time, and synchronized planned and unplanned events with their subtypes and durations. The dataset is highly imbalanced, with only 0.87% of {tmc, time_bin} entries containing any event information. The enriched hourly dataset is stored at `/database/i10-broadway/X_full_1h.parquet`, and the segment-level travel time can be visualized below.

![true](images/heatmap_truth_WB.png)
*Travel time heatmap by segments on Westbound I-10 Broadway Curve from June-October 2025.*

## Exploratory Data Analysis
Our exploratory analysis confirms the reliability of INRIX traffic data, but found some reporting issues with the AZ511 event data. Feature importance analysis is also attached.

![match_events](images/i10_broadway_tmc.png)
*Spatial matching of AZ511 events to TMC segments along the I-10 Broadway Curve. Each color represents a unique TMC segment.*

#### Traffic pattern
 We see no major issue with INRIX data. Traffic patterns are clear: Weekday traffic shows morning and afternoon slowdowns corresponding to commuter peaks, while weekends maintain consistently higher speeds throughout the day. Seasonally, average speeds are lower during the winter months (due to increased travel demand from seasonal visitors) and higher during the summer, when overall traffic volumes are lighter.

![traffic-speed-1yr](images/speed_1y.png)
*Daily average traffic speed along the I-10 Broadway Curve over one year.*

![traffic-speed-heatmap](images/speed_dow.png)
*Average traffic speed (mph) by hour and day of week along the I-10 Broadway Curve.*

#### Event reporting issue
Upon reviewing AZ511 event records, we found that updates appear to occur in batch intervals of ~3 hours, and unplanned events are significantly underreported. Several crashes reported in local news were missing from AZ511, which suggests gaps in real-time data capture.

![evt-heatmap](images/reported_accidents_dow.png)
*Count of all events aggregated by day of week and hour of day*

While some event descriptions contain detailed narratives, the EventSubType field is inconsistently defined, with vague or nonstandard entries (e.g., C34Rshoulder instead of “Crash on right shoulder,” and undefined codes such as T1018). Furthermore, the Severity field is missing in roughly 90% of records, limiting its usefulness for feature engineering. To ensure analytical consistency, we manually reclassified events into two categories—planned (e.g., work zones, closures) and unplanned (e.g., crashes, incidents)—based on their subtype descriptions.

![evt-heatmap](images/severityxeventsubtype.png)

*Distribution of events by subtype and reported severity*

#### Feature Importance
In addition, we explored feature importance by fitting an XGBoost model (see [Modeling Approach](#modeling-approach)). Result show that lagged travel-time (up to 3 hours) feature dominates model performance, while event-related features contribute the least, which can be another evidence that event reporting may not be in sync with the resulting traffic patterns. The ranking below is based on feature significance (p-values).
![fi](images/xgb_full_feature_importance_nested_pie.png)
*Feature importance identified by XGBoost*


## Modeling Approach
We trained two classes of models based on their treatment of spatial and temporal dependencies. Tabular regression models—including Linear Regression and Tree-Based models—were trained on independent {tmc, time_bin} entries without explicit spatial or temporal structure. In contrast, sequence models such as Long Short-Term Memory (LSTM) and Graph Convolutional Network LSTM (GCN-LSTM) were designed to capture temporal continuity and, in the case of GCN-LSTM, spatial correlations between connected TMC segments. We then compare the traditional regression baselines with the deep learning models.

The best-performing model and feature combination were selected to conduct a counterfactual analysis, enabling estimation of event-induced delay in the absence of a direct control group (i.e., no-event conditions). Specifically, the control scenario was modeled by removing all data within ±3 hours of reported event times and excluding event-related and lag features that could be influenced by those events. Because this process disrupts temporal continuity, sequence models were not used, as they require complete, continuous time-series data for training.

### Model Overview
|Model Class| Model Family | Description | Training Data
|----|----------|-------------|-------------|
|Tabular regression models | **Linear Regression** | Ordinary Least Squares (OLS), Ridge and Lasso. Assume linear relationships between events, time features, and travel time. No interaction terms. No spatial or temporal dependency.| Event-balanced data points[^1] |
|Tabular regression models | **Tree-Based Models** | Random Forest (rf), Gradient Boosted Regression Trees (gbrf) and XGBoost. Captures nonlinearities and interactions, No spatial or temporal dependency.|Event-balanced data points[^1] |
|Sequence models | **LSTM** | A recurrent neural network for time-series prediction. A global (pooled) model trained on all TMC time-series sliced into short 24-hr sequences. Consider temporal but no spatial dependency.| Sliced short sequences across all TMCs |
|Sequence models| **GCN-LSTM** | A spatial-temporal graph neural network (ST-GNN) model. GCN captures spatial dependencies between conneted TMCs, and LSTM learns temporal dynamics| Sliced short sequences across all TMCs |
<!-- | **SARIMAX Models** | Seasonal ARIMA with Exogenous Regressors. Models serial dependence and seasonality directly, trained independently for each TMC. Computationally heavy. Consider temporal but no spatial dependency. | Full time-series for each TMC | -->

[^1]: The training data is a multi-index DataFrame with indices {tmc, time_bin}, where events are counted at each entry. Since events occur in less than 1% of 1-hr time bins, we downsample non-event entries to achieve approximately 50% event balance in the training data.

### Feature choices
| Features | Description | 
|----------|-------------|
| **Road**  `(miles, on/off ramps, curve)`| Static features related to road geometry each TMC. Some are manually tagged from Google Satellite View |
| **Events (evt)**  `(evt_cat_planned, evt_cat_unplanned)`| Counts or presence of planned events (closures, roadwork, etc.) and unplanned events (crashes, debris, accidents etc.)|
|**Cyclic time (cyc)** `(hour_sin, hour_cos, dow_sin, dow_cos, hour_of_week_sin, hour_of_week_cos, is_weekend)`|Encodes daily & weekly periodicity|
|**Lagged travel time (lag)** `(travel_time_t-1, t-2,...)`|Captures short-term persistence|
<!-- |**Seasonality** `(P,D,Q,s)`|Explicit periodic autocorrelation|SARIMAX only| -->

For each model family in Tabular models, various regressor combinations were tested. The configurations include:
1. road features only
2. road + evt
3. road + evt + lag 
4. road + lag 
5. road + cyc
6. road + cyc + lag
7. full features: road + cyc + evt + lag



## Results

### Model training results
![true](images/tabular_models_cv_rmse.png)

*Quick model comparison by CV RMSE*: adding "lag" feature improved the accuracy for all models; "XGBoost" is the overall best model.

![true](images/full_feature_models_cv_rmse.png)

*Full-feature model comparison by test RMSE*: LSTM has the best prediction accuracy, followed by XGBoost.

The following travel-time heatmaps visualize the model prediction results using "full" features for XGBoost and LSTM.
<!-- ![true](images/heatmap_truth_WB.png)
*True travel time heatmap* -->

<!-- ![lr](images/heatmap_lr_full_WB.png)
*Linear Regression with full features*

![gbrt](images/heatmap_gbrt_full_WB.png)
*Gradient boosted tree with full features* -->

![xgb](images/heatmap_xgb_full_WB.png)
*XGBoost model with full features*

![lstm](images_old/heatmap-lstm-full.png)
*LSTM model with full features*

<!-- ![sarimax](images/sarimax_full.png)
*SARIMAX model with full features* -->

<!-- ![gcn](images/heatmap_gcn_WB.png)
*GCN-LSTM predicted travel time heatmap* -->



**! Important Note:** All models are trained to make one-shot prediction, i.e., given current or some lagged states, predict the travel time in the next time_bin. They are not designed for multi-step or recursive forecasting.


### Event-induced delay
Event-induced delay was estimated using a counterfactual prediction model trained with XGBoost excluding all event-related data and features, allowing isolation of delay attributable to work zones and incidents. The analysis focused on planned events (e.g., roadwork), as they are typically longer in duration and less affected by underreporting.

The additional delay was computed as the difference between travel times predicted by the full model and the counterfactual “no-event” model. Results indicate that event-related delays peak during early afternoons (12–3 PM) on weekdays and in the mornings on weekends. While weekend or overnight closures—common DOT mitigation strategies—help reduce congestion, our findings show that these measures reduce but do not eliminate delays, particularly on weekends.

![delay-how](images/extra_delay_by_hour.png)

![delay-how](images/extra_delay_heatmap_dow_hour.png)

Further analysis reveals that planned work zones cause prolonged, lagged effects, with delays peaking 3–5 hours after work begins and reaching up to 15 sec/mile (2-3min extra delay on the 11-mile corridor), which doubles the typical daily congestion delay.

![delay-how](images/diff_by_event_type_boxplot.png)

![delay-how](images/corr_evt_delay.png)

These results are consistent with ADOT’s observations that off-peak closures lessen congestion but cannot eliminate it entirely. The observed lagged and residual delays also underscore the need for more complete and timely event reporting to improve predictive accuracy and mitigation planning.

## Future Work
1. Integrate real-time incident feeds to address underreported and delayed event updates, such as  exploring connected vehicle data or roadside sensor information (e.g., CCTV camera).
2. Expand the feature context by incorporating weather conditions and detailed event descriptions.
3. Explore causal inference methods to rigorously estimate the delay impact attributable to multiple interacting factors such as work zones, demand fluctuations, and incidents.

## Quick Start
Before running the following commands, make sure to follow the project structure specified in [Description of Repository](#description-of-repository), and have 
- `az511.db` stored in `database/`, and
- raw INRIX data stored as `database/inrix-traffic-speed/I10-and-I17-1year/`

Run the entire pipeline end-to-end using the following commands:

```bash
# 1) Prepare dataset (events + INRIX -> parquet)
#    Output: database/i10-broadway/X_full_1h.parquet
#    - Final feature table with MultiIndex {tmc, time_bin}
python prepare_i10_training_data.py

# 2) Train tabular baseline models (Linear/Ridge/Lasso, RF/GBRT, XGBoost)
#    Output: models/tabular_run/
python train_model_tabular.py --save-models

# 3) Train sequence model (LSTM)
#    Output: models/lstm_run/
python train_model_lstm.py --save-models

# 4) Train spatial-temporal model (GCN + LSTM)
#    Output: models/gcn/gcn_lstm_i10_wb/
python train_model_stgnn.py

# 5) Compare model performance and generate evaluation figures
#    Output: images/
python model_comparison.py --direction WB --save-figs

# 6) Run counterfactual analysis to estimate delay impact of planned events
#    Output: images/
python evt_impact_analysis.py
```



<!-- ## Data Dashboard

![Work Zone Dashboard](images/workzone.png)
*Interactive map showing AZ511 work zones and traffic data across Arizona*

- **AZ511 Work Zone Monitoring**
  - Real-time work zone events
  - Construction and incident tracking
  - Geographic distribution analysis
  - Duration and timing analytics


- **Interactive Visualizations**
  - Combined map view with work zones and traffic flow
  - City-based filtering (Phoenix, Tucson, Flagstaff, Gilbert, Yuma)
  - Date range selection
  - Event type distribution charts
  - Duration analysis histograms


## Dashboards
**Dashboard to visualize events and traffic**:
```bash
streamlit run dashboard/az511app.py
```

Dashboard Features:
- **Data Source Selection**: Toggle between AZ511 work zones and TomTom traffic flow
- **Geographic Filtering**: Filter by city (Phoenix, Tucson, Flagstaff, Gilbert, Yuma)
- **Date Selection**: View data for specific dates
- **Interactive Map**: 
  - AZ511 events shown as colored markers (by event type)
  - TomTom traffic flow displayed as colored polylines with 5-tier speed system:
    - Green: Excellent flow (90%+ free flow speed)
    - Light Green: Good flow (70-90%)
    - Yellow: Moderate flow (50-70%)
    - Orange: Slow flow (30-50%)
    - Red: Very slow/stopped (<30%)
  - Arizona road network overlay from GeoJSON files (interstates and state routes)
  - Functional Road Class (FRC) filtering with performance optimizations
  - Smart rendering limits for local roads (FRC4+ limited to 1000 segments)
  - Legend positioned on the right side of the map
- **Analytics Charts**: 
  - Event type distribution
  - Work zone duration analysis
  - Update vs start date patterns

![Analytics Dashboard](images/analytics.png)
*Comprehensive analytics including event distributions, duration analysis, and temporal patterns* -->

## Description of Repository
The project repository is organized as follows. 
```
wzdx/
├── environment.yml              # Conda environment specification
├── requirements.txt             # Python dependencies (pip)
├── run_az511.sh                 # Shell script to run AZ511 job
├── README.md                    # Project documentation
├── _log/                        # Logs from data collection
│   └── az511_28538440.err
├── dashboard/                   # Streamlit dashboard applications
│   ├── az511app.py              # AZ511 work zone + traffic dashboard
│   ├── inrixapp.py              # INRIX-specific dashboard
│   ├── wzdxapp.py               # WZDx-focused dashboard
│   └── __pycache__/             # Bytecode cache
├── database/                    # Data files, scripts, and assets
│   ├── az511.py                 # AZ511 data collection script
│   ├── az511.db                 # AZ511 data
│   ├── wzdx.py                  # Work zone data processing script
│   ├── i10-broadway/            # Processed training data
│   │   ├── X_tensor_1h.npz
│   └── inrix-traffic-speed/     # Raw data from INRIX (not shared)
│       └── I10-and-I17-1year/
│           ├── Contents.txt
│           ├── I10-and-I17-1year.csv
│           └── TMC_Identification.csv
├── images/                      # Figures for README and dashboards
├── models/                      # Model training output
│   └── gcn/                     # Training results for GCN-LSTM
│   └── lstm_run/                # Training results for LSTM
│   └── tabular_run/             # Training results for LR and Tree-based models using tabular data
├── notebooks/                   # EDA and adhoc scripts
├── src/                         # Generic helper utilities
├── evt_impact_analysis.py       # Script to 
├── model_comparison.py          # Script to
├── prepare_i10_training_data.py # Script to prepare the dataset (events + INRIX -> parquet)
├── train_model_lstm.py          # Script to train LSTM
├── train_model_stgnn.py         # Script to train ST‑GNN
└── train_model_tabular.py       # Script to train tabular baselines (Linear/Ridge/Lasso, RF/GBRT, XGBoost)
```

<!-- ## Data Access

This project uses a combination of proprietary and public datasets:

- **INRIX data** — Licensed and proprietary. The raw data cannot be shared due to contractual restrictions.  
  However, **processed and aggregated training data** (e.g., anonymized feature tables, model inputs, or summary statistics) can be shared upon reasonable request.  

- **AZ511 data** — Publicly available through the [Arizona Department of Transportation (ADOT) 511 API](https://www.az511.com/).  
  You can access it directly by registering for an API key or using their open endpoints.

Processed datasets and derived features included in this repository are shared under the same license as the code (MIT), unless otherwise noted.

For questions or data-sharing inquiries, please contact
**Yanbing Wang**  -->

## License
MIT License

Copyright (c) 2025 Yanbing Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.