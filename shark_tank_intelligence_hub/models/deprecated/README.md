# Deprecated Models

## equity_predictor_regression.pkl

**Status:** DEPRECATED - DO NOT USE

**Reason:** Severe data leakage detected
- Achieved R² = 0.976 (unrealistically high)
- Used post-deal features (Total Deal Equity, etc.)
- Equity dilution can only be known AFTER deal is made
- Not useful for pre-pitch prediction

**Replacement:** None - equity dilution prediction removed from project scope

**Date Deprecated:** February 16, 2026
