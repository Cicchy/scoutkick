# ScoutKick - EPA Model Port for FTC

## Project Goal
Implement a standalone Expected Points Added (EPA) rating system for FIRST Tech Challenge (FTC), porting and adapting the mathematical foundations of the Statbotics FRC model.

## Architecture (Statbotics-Inspired)
The project follows a decoupled Client-Server architecture to allow for independent scaling and development of the data engine and the presentation layer.

### 📂 Repository Structure
- **`backend/`**: The core logic, data pipeline, and API.
    - `src/core/`: Mathematical foundations (e.g., `math.py` for Skew-Normal distributions).
    - `src/data/`: ETL layer. Transforms raw match data into EPA vectors (e.g., `cleaner.py`).
    - `src/services/`: Orchestration layer. Implements the EPA loop (e.g., `epa_service.py`).
    - `src/storage/`: Database interfaces for persistence (Planned).
    - `src/api/`: REST/GraphQL interfaces for the frontend (Planned).
- **`frontend/`**: The presentation layer (Next.js/Svelte) (Planned).
- **`tests/`**: Verification scripts and unit tests (e.g., `test_engine.py`).

---

## Inner Workings & Adaptations

### 1. The FTC EPA Vector (2025)
The model operates on a standardized vector instead of raw API objects.
**Indices:** `[0: Total, 1: Auto, 2: Teleop, 3: Endgame, 4: RP1, 5: RP2, 6: RP3, 7: Auto_Class, 8: Tele_Class, 9: Tele_Depot]`
- Total points use `preFoulTotal` as the baseline.
- Ranking Points (RP) are treated as binary (0 or 1).

### 2. Alliance Logic (FRC $\rightarrow$ FTC)
- **Team Count**: Adjusted `num_teams` from 3 (FRC) to 2 (FTC).
- **Attribution**: Error is split across 2 teams instead of 3, increasing the impact of individual team performance on their rating.

### 3. Probability Modeling
- **Win Probability**: Uses a logistic function normalized by the season's `score_sd`.
- **RP Prediction**: Predicted RP sums are passed through a `unit_sigmoid` to map them to a $0.0 \rightarrow 1.0$ probability range.

### 4. The Learning Engine (EWMA)
- **Distribution**: Teams are represented as `SkewNormal` distributions (Mean, Variance, Skewness).
- **Learning Rate ($\alpha$)**: Implements a decaying learning rate: $\alpha = 1 / (1 + n \times 0.1)$.
- **Weighting**: Supports match weighting (e.g., Elimination matches can be set to a lower weight like $0.33$).

---

## Future Roadmap & Polish

### Priority: High (Critical for Production)
- [ ] **Data Persistence**: Implement a database layer in `backend/src/storage/` to save team distributions across sessions.
- [ ] **Automatic Calibration**: Create a utility to calculate the actual `score_sd` from a dataset of matches to refine win probabilities.

### Priority: Medium (Quality of Life)
- [ ] **Cross-Season Initialization**: Implement Mean Reversion to carry over ratings from previous seasons (e.g., 2024 $\rightarrow$ 2025).
- [ ] **Dynamic Dimensions**: Allow the `Cleaner` to adapt to different seasons without manually rewriting the vector indices.

### Priority: Low (Precision Tuning)
- [ ] **Component-Based Attribution**: Instead of splitting error equally, attribute points based on which team actually scored the specific component.
- [ ] **Advanced Skewness**: Refine the skewness update formula to better capture "boom/bust" team behavior.
