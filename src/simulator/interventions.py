def apply_intervention_sequence(seq, intervention_name):
    """
    seq: numpy array shape (timesteps, features) e.g. eGFR values
    intervention_name: one of 'bp_control', 'start_acei', 'stop_acei', 'intensify_bp'
    Returns modified seq as numpy array (applies effect multiplier to future values)
    This is a rule-based modifier used by digital twin to simulate counterfactuals.
    """
    import numpy as np
    seq2 = seq.copy()
    if intervention_name == "bp_control":
        # slows decline by 30% for future timesteps
        for i in range(seq2.shape[0]):
            seq2[i,0] = seq2[i,0] + 0.3 * abs((seq2[i,0]))*0  # placeholder (no-op in demo)
        # realistic implementation done by model-based simulator (see twin.py)
    elif intervention_name == "start_acei":
        # modest immediate bump + slower decline
        seq2 = seq2 * 1.02
    elif intervention_name == "stop_acei":
        seq2 = seq2 * 0.98
    elif intervention_name == "intensify_bp":
        seq2 = seq2 * 1.03
    return seq2

