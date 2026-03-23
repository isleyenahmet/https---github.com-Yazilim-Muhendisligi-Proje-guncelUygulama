import sqlite3
import pandas as pd

def analyze_db():
    try:
        conn = sqlite3.connect('aiops_stream.db')
        
        # Get tables
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        print(f"Veritabanındaki Tablolar: {[t[0] for t in tables]}")
        
        if tables:
            for t in tables:
                t_name = t[0]
                
                # Get total rows
                count = conn.execute(f"SELECT COUNT(*) FROM {t_name};").fetchone()[0]
                
                # Get column names and sample data
                df = pd.read_sql_query(f"SELECT * FROM {t_name} LIMIT 5", conn)
                
                print(f"\n--- {t_name} Tablosu Analizi ---")
                print(f"Toplam Satir Sayisi: {count:,}")
                print(f"Toplam Sutun Sayisi: {len(df.columns)}")
                print(f"Sutun Isimleri: {list(df.columns)}\n")
                
                print("Ilk 5 Satir Ornegi:")
                print(df.head().to_string())
                
                # Sütun veri tipleri ve eksik veri kontrolü(örnek)
                print("\nVeri Tipleri ve Ozet (Limitsiz bir kesit uzerinden):")
                df_sample = pd.read_sql_query(f"SELECT * FROM {t_name} LIMIT 1000", conn)
                print(df_sample.describe().to_string())
                
        conn.close()
    except Exception as e:
        print(f"Hata oluştu: {e}")

if __name__ == "__main__":
    analyze_db()
