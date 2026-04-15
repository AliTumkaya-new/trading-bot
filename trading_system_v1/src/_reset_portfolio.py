"""Portföyü sıfırla — temiz başlangıç."""
import sys
sys.path.insert(0, ".")
from database.db import init_db, _conn

init_db()
with _conn() as con:
    con.execute("DELETE FROM positions")
    con.execute("DELETE FROM trades")
    con.execute("DELETE FROM signals_log")
    con.execute("DELETE FROM daily_snapshots")
    con.execute("DELETE FROM portfolio")
    # ML tabloları varsa
    try:
        con.execute("DELETE FROM indicator_performance")
    except Exception:
        pass
    try:
        con.execute("DELETE FROM learned_weights")
    except Exception:
        pass

print("Tüm veriler silindi. Portföy sıfırlandı.")
print("Bir sonraki 'python main.py --once' çağrısında ₺5,000 ile temiz başlanacak.")
