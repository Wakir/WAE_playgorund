from sklearn.metrics import recall_score

def specificity(y_true, y_pred):
    """
    Oblicza specyficzność (specificity) na podstawie etykiet prawdziwych i przewidywanych.

    Args:
        y_true (array-like): Rzeczywiste etykiety klas.
        y_pred (array-like): Przewidywane etykiety klas.

    Returns:
        float: Wartość specyficzności.
    """
    # Używamy recall_score z pos_label=0 do obliczenia specificity
    return recall_score(y_true, y_pred, pos_label=0)