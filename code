import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================

def load_data(max_rows=150000):
    files = {
        'Monday': 'Monday-WorkingHours.pcap_ISCX.csv',
        'Tuesday': 'Tuesday-WorkingHours.pcap_ISCX.csv',
        'Wednesday': 'Wednesday-workingHours.pcap_ISCX.csv',
        'Thursday_Infiltration': 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        'Friday_PortScan': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        'Friday_DDoS': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        'Friday_Botnet': 'Friday-WorkingHours-Morning.pcap_ISCX.csv'
    }
    
    rows_per_file = max_rows // len(files)
    all_dfs = []
    
    for name, path in files.items():
        try:
            df = pd.read_csv(path, low_memory=False)
            df.columns = df.columns.str.strip()
            if len(df) > rows_per_file:
                df = df.sample(n=rows_per_file, random_state=42)
            print(f"Загружен {name}: {len(df)}")
            all_dfs.append(df)
        except Exception as e:
            print(f"Ошибка {name}: {e}")
    
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all[~df_all.isna().all(axis=1)]
    df_all = df_all.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    y = (df_all['Label'] != 'BENIGN').astype(int)
    X = df_all.drop('Label', axis=1).select_dtypes(include=[np.number])
    X = X.loc[:, X.var() != 0]
    
    print(f"Признаков: {X.shape[1]}, Атак: {y.sum()}, Норма: {len(y)-y.sum()}")
    return X, y

# ============================================
# 2. ДИСКРЕТИЗАЦИЯ
# ============================================

def discretize_by_median(X_train, X_data, feature_names):
    """Дискретизация по медиане обучающей выборки"""
    medians = {}
    for col in feature_names:
        medians[col] = X_train[col].median()
    
    X_bin = []
    for i in range(len(X_data)):
        row = []
        for col in feature_names:
            val = X_data.iloc[i][col]
            median = medians[col]
            row.append(f"{col}_low" if val <= median else f"{col}_high")
        X_bin.append(row)
    
    return pd.DataFrame(X_bin, columns=feature_names), medians

# ============================================
# 3. ОБУЧЕНИЕ ДВУХ МЕТОДОВ
# ============================================

def train_two_methods(X_train, y_train):
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=-1)
    
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    
    return dt, rf

# ============================================
# 4. ПРИМЕНЕНИЕ ПРАВИЛ
# ============================================

def apply_rules_to_test(rules, X_test_bin):
    """Проверяет, какие правила сработали"""
    if len(rules) == 0:
        return np.zeros(len(X_test_bin))
    
    attack_rules = []
    for idx, rule in rules.iterrows():
        ants = set(rule['antecedents'])
        cons = list(rule['consequents'])
        if any('label_attack' in str(c) for c in cons):
            attack_rules.append(ants)
    
    triggered = np.zeros(len(X_test_bin))
    
    for i in range(len(X_test_bin)):
        row_set = set()
        for col in X_test_bin.columns:
            row_set.add(X_test_bin.iloc[i][col])
        
        for rule_set in attack_rules:
            if rule_set.issubset(row_set):
                triggered[i] = 1
                break
    
    return triggered

# ============================================
# 5. ОСНОВНАЯ ФУНКЦИЯ
# ============================================

def run_hybrid_with_rules():
    print("="*70)
    print("ГИБРИДНЫЙ МЕТОД: ДВА МЕТОДА + АССОЦИАТИВНЫЕ ПРАВИЛА")
    print("="*70)
    
    # Загрузка
    print("\n[1/5] Загрузка...")
    X, y = load_data(max_rows=150000)
    
    # Разделение
    print("\n[2/5] Разделение...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение
    print("\n[3/5] Обучение Decision Tree и Random Forest...")
    dt, rf = train_two_methods(X_train_scaled, y_train)
    
    # Получаем вероятности
    proba_dt = dt.predict_proba(X_test_scaled)[:, 1]
    proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

    # Мягкое голосование (soft voting) — среднее вероятностей
    proba_ensemble = (proba_dt + proba_rf) / 2

   # Ансамблевая метка по порогу 0.5
   y_pred_ensemble = (proba_ensemble >= 0.5).astype(int) 
   
    # Метрики ансамбля
    cm = confusion_matrix(y_test, y_pred_ensemble)
    tn, fp, fn, tp = cm.ravel()
    fpr_ensemble = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"\nАнсамбль (DT + RF): FP={fp}, FN={fn}, FPR={fpr_ensemble:.6f}")
    
    # Находим ошибки (FP и FN)
    error_indices = ((y_test == 0) & (y_pred_ensemble == 1)) | ((y_test == 1) & (y_pred_ensemble == 0))
    X_errors = X_test.iloc[error_indices.values]
    y_errors = y_test.iloc[error_indices.values]
    print(f"\nОшибок для генерации правил: {len(X_errors)}")
    
    if len(X_errors) < 10:
        print("Слишком мало ошибок")
        return
    
    # Дискретизация
    print("\n[4/5] Дискретизация...")
    feature_names = X.columns.tolist()
    X_errors_bin, medians = discretize_by_median(X_train, X_errors, feature_names)
    
    # Добавляем метку
    df_errors = X_errors_bin.copy()
    df_errors['label'] = y_errors.values
    df_errors['label'] = df_errors['label'].apply(lambda x: 'attack' if x == 1 else 'normal')
    
    # One-hot кодирование
    df_encoded = pd.get_dummies(df_errors)
    print(f"  Столбцов после one-hot: {df_encoded.shape[1]}")
    
    # Apriori
    print("\n[5/5] Генерация ассоциативных правил...")
    df_encoded = df_encoded.astype(bool)
    min_support = max(0.03, 3.0 / len(df_errors))
    print(f"  min_support = {min_support:.4f}")
    
    frequent = apriori(df_encoded, min_support=min_support, use_colnames=True, max_len=2, low_memory=True)
    
    if len(frequent) > 0:
        rules = association_rules(frequent, metric="confidence", min_threshold=0.5)
        rules = rules[rules['consequents'].apply(lambda x: any('label_' in str(i) for i in x))]
        print(f"  Сгенерировано правил с меткой: {len(rules)}")
        
        if len(rules) > 0:
            print("\n  ПРИМЕРЫ ПРАВИЛ:")
            for i, (idx, row) in enumerate(rules.head(10).iterrows()):
                ants = list(row['antecedents'])[:3]
                cons = list(row['consequents'])
                print(f"    {i+1}. {ants} → {cons}")
                print(f"        lift={row['lift']:.2f}, conf={row['confidence']:.2f}")
            
            # Применяем правила к тестовой выборке
            print("\n  Применение правил к тестовой выборке...")
            X_test_bin, _ = discretize_by_median(X_train, X_test, feature_names)
            triggered = apply_rules_to_test(rules, X_test_bin)
            
            # Вероятности ансамбля
            proba_dt = dt.predict_proba(X_test_scaled)[:, 1]
            proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
            proba_ensemble = (proba_dt + proba_rf) / 2
            
            # Гибридные предсказания: если правило сработало -> атака, иначе по ансамблю
            y_test_hybrid = []
            for i in range(len(X_test)):
                if triggered[i]:
                    y_test_hybrid.append(1)
                else:
                    y_test_hybrid.append(1 if proba_ensemble[i] >= 0.5 else 0)
            y_test_hybrid = np.array(y_test_hybrid)
            
            # Оценка гибридного метода
            cm_hybrid = confusion_matrix(y_test, y_test_hybrid)
            tn_h, fp_h, fn_h, tp_h = cm_hybrid.ravel()
            fpr_hybrid = fp_h / (fp_h + tn_h) if (fp_h + tn_h) > 0 else 0
            
            print(f"\n  РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ:")
            print(f"    Ансамбль (без правил):        FPR = {fpr_ensemble:.6f} ({fp} FP)")
            print(f"    Гибридный метод (с правилами): FPR = {fpr_hybrid:.6f} ({fp_h} FP)")
            
            if fpr_hybrid < fpr_ensemble:
                reduction = (fpr_ensemble - fpr_hybrid) / fpr_ensemble * 100
                print(f"\n     СНИЖЕНИЕ FPR: {reduction:.1f}%")
                print(f"     Сокращение FP: {fp} → {fp_h} (-{fp - fp_h})")
            else:
                print(f"\n     УВЕЛИЧЕНИЕ FPR")
        else:
            print("  Нет правил с меткой")
    else:
        print("  Нет частых наборов")

# ============================================
# ЗАПУСК
# ============================================

if __name__ == "__main__":
    run_hybrid_with_rules()    
