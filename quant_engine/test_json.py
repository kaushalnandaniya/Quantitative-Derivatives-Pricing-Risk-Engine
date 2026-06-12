import json
import numpy as np
from fastapi.encoders import jsonable_encoder

results = {'total_trades': 31, 'wins': 0, 'losses': 31, 'win_rate': 0.0, 'total_pnl': 0.0, 'avg_pnl': np.float64(0.0), 'profit_factor': np.inf}

try:
    encoded = jsonable_encoder(results)
    json_str = json.dumps(encoded)
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
