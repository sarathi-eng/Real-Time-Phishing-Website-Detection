def explain_decision(model, feature_vector: dict) -> list:
    """
    Decision Explainability (XAI) Module.
    Extracts the underlying ML math to identify the Top 3 Features 
    that influenced the Random Forest's evaluation arrays.
    """
    import numpy as np
    try:
        # CalibratedClassifierCV stores fitted clones inside calibrated_classifiers_
        classifiers = model.model.calibrated_classifiers_
        # Average the weights across all 3 CV folds
        importances = np.mean([clf.estimator.feature_importances_ for clf in classifiers], axis=0)
    except AttributeError:
        return ["Feature importance currently unmapped or model unfitted."]
        
    feature_names = list(feature_vector.keys())
    
    # Sort features mathematically by their node entropy contribution
    feat_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    top_3 = []
    for name, imp in feat_importances[:3]:
        contribution = round(imp * 100, 1)
        # Identify the exact value that triggered this request's logic
        val = feature_vector[name]
        top_3.append(f"Feature '{name}' (Value: {val}) contributed {contribution}% to the ML logic.")
        
    return top_3
