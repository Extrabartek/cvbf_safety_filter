def read_tpf(filename):
    """
    Read tire property file (.tir/.tpf) and extract parameters

    Parameters:
    -----------
    filename : str
        Path to the .tir or .tpf file

    Returns:
    --------
    dict : Dictionary containing all tire parameters
    """

    # Read the file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find the [MODEL] section
    model_idx = None
    for i, line in enumerate(lines):
        if '[MODEL]' in line:
            model_idx = i
            break

    if model_idx is None:
        raise ValueError(f"Could not find [MODEL] section in {filename}")

    # Extract text after [MODEL]
    text_tyre = lines[model_idx:]

    # Initialize tyre dictionary with default values
    tyre = {}

    # Helper function to read parameter with default value
    def readcheck(text, param_name, default_value=0):
        for line in text:
            if param_name in line and '=' in line and not line.strip().startswith('!'):
                # Extract value after '='
                parts = line.split('=')
                if len(parts) >= 2:
                    # Remove comments and whitespace
                    value_str = parts[1].split('$')[0].strip()
                    value_str = value_str.replace("'", "").strip()
                    try:
                        # Handle special string values
                        if param_name in ['TYRESIDE', 'FITTYP']:
                            return value_str if value_str else default_value
                        # Convert to float for numeric values
                        return float(value_str)
                    except:
                        return default_value
        return default_value

    # MODEL section
    tyre['fittyp'] = readcheck(text_tyre, 'FITTYP', 62)
    tyre['tyreside'] = readcheck(text_tyre, 'TYRESIDE', 0)
    tyre['LONGVL'] = readcheck(text_tyre, 'LONGVL', 100)
    tyre['Vxlow'] = readcheck(text_tyre, 'VXLOW', 1)

    # DIMENSION section
    tyre['R_0'] = readcheck(text_tyre, 'UNLOADED_RADIUS', 0)
    tyre['width'] = readcheck(text_tyre, 'WIDTH', 0.00001)
    tyre['Rim_R'] = readcheck(text_tyre, 'RIM_RADIUS', 0)
    tyre['Rim_width'] = readcheck(text_tyre, 'RIM_WIDTH', 0)
    tyre['AR'] = readcheck(text_tyre, 'ASPECT_RATIO', 0.6)

    # INERTIA section
    tyre['mass'] = readcheck(text_tyre, 'MASS', 12.583765)
    tyre['Ixx'] = readcheck(text_tyre, 'IXX', 1.034574)
    tyre['Iyy'] = readcheck(text_tyre, 'IYY', 2.19194)

    # Vertical coefficients
    tyre['F_z0'] = readcheck(text_tyre, 'FNOMIN', 5000)
    tyre['C_z0'] = readcheck(text_tyre, 'VERTICAL_STIFFNESS', 1930.735546)
    tyre['K_z0'] = readcheck(text_tyre, 'VERTICAL_DAMPING', 50)
    tyre['contourA'] = readcheck(text_tyre, 'MC_CONTOUR_A', 0)
    tyre['contourB'] = readcheck(text_tyre, 'MC_CONTOUR_B', 0)
    tyre['Breff'] = readcheck(text_tyre, 'BREFF', 8)
    tyre['Dreff'] = readcheck(text_tyre, 'DREFF', 0.24)
    tyre['Freff'] = readcheck(text_tyre, 'FREFF', 0.01)
    tyre['q_re0'] = readcheck(text_tyre, 'Q_RE0', 1)
    tyre['q_v1'] = readcheck(text_tyre, 'Q_V1', 0)
    tyre['q_v2'] = readcheck(text_tyre, 'Q_V2', 0)
    tyre['q_Fz2'] = readcheck(text_tyre, 'Q_FZ2', 0)
    tyre['q_Fcx'] = readcheck(text_tyre, 'Q_FCX', 0)
    tyre['q_Fcy'] = readcheck(text_tyre, 'Q_FCY', 0)
    tyre['q_Fcy2'] = readcheck(text_tyre, 'Q_FCY2', 0)
    tyre['q_cam'] = readcheck(text_tyre, 'Q_CAM', 0)
    tyre['q_cam1'] = readcheck(text_tyre, 'Q_CAM1', 0)
    tyre['q_cam2'] = readcheck(text_tyre, 'Q_CAM2', 0)
    tyre['q_cam3'] = readcheck(text_tyre, 'Q_CAM3', 0)
    tyre['q_fys1'] = readcheck(text_tyre, 'Q_FYS1', 0)
    tyre['q_fys2'] = readcheck(text_tyre, 'Q_FYS2', 0)
    tyre['q_fys3'] = readcheck(text_tyre, 'Q_FYS3', 0)
    tyre['PFZ1'] = readcheck(text_tyre, 'PFZ1', 0)
    tyre['bottom_offst'] = readcheck(text_tyre, 'BOTTOM_OFFST', 0)
    tyre['bottom_stiff'] = readcheck(text_tyre, 'BOTTOM_STIFF', 9653.677)

    # Structural coefficients
    tyre['Cx0'] = readcheck(text_tyre, 'LONGITUDINAL_STIFFNESS', 317750)
    tyre['Cy0'] = readcheck(text_tyre, 'LATERAL_STIFFNESS', 120100)
    tyre['Cyaw'] = readcheck(text_tyre, 'YAW_STIFFNESS', 0)
    tyre['d_residual'] = readcheck(text_tyre, 'DAMP_RESIDUAL', 0.002)
    tyre['d_vlow'] = readcheck(text_tyre, 'DAMP_VLOW', 0.00125)
    tyre['PCFX1'] = readcheck(text_tyre, 'PCFX1', 0)
    tyre['PCFX2'] = readcheck(text_tyre, 'PCFX2', 0)
    tyre['PCFX3'] = readcheck(text_tyre, 'PCFX3', 0)
    tyre['PCFY1'] = readcheck(text_tyre, 'PCFY1', 0)
    tyre['PCFY2'] = readcheck(text_tyre, 'PCFY2', 0)
    tyre['PCFY3'] = readcheck(text_tyre, 'PCFY3', 0)
    tyre['PCMZ1'] = readcheck(text_tyre, 'PCMZ1', 0)

    # Operating limits
    tyre['p_lim'] = [readcheck(text_tyre, 'PRESMIN', 100000), readcheck(text_tyre, 'PRESMAX', 300000)]
    tyre['p_i0'] = readcheck(text_tyre, 'INFLPRES', 200000)
    tyre['p_nom'] = readcheck(text_tyre, 'NOMPRES', 200000)
    tyre['F_z'] = [readcheck(text_tyre, 'FZMIN', 1), readcheck(text_tyre, 'FZMAX', 150000)]
    tyre['kappa'] = [readcheck(text_tyre, 'KPUMIN', -1.5), readcheck(text_tyre, 'KPUMAX', 1.5)]
    tyre['alpha'] = [readcheck(text_tyre, 'ALPMIN', -1.58), readcheck(text_tyre, 'ALPMAX', 1.58)]
    tyre['gamma'] = [readcheck(text_tyre, 'CAMMIN', -0.5), readcheck(text_tyre, 'CAMMAX', 0.5)]

    # Scaling coefficients
    tyre['LFZ0'] = readcheck(text_tyre, 'LFZO', 1)
    tyre['LCX'] = readcheck(text_tyre, 'LCX', 1)
    tyre['LMUX'] = readcheck(text_tyre, 'LMUX', 1)
    tyre['LEX'] = readcheck(text_tyre, 'LEX', 1)
    tyre['LKX'] = readcheck(text_tyre, 'LKX', 1)
    tyre['LHX'] = readcheck(text_tyre, 'LHX', 1)
    tyre['LVX'] = readcheck(text_tyre, 'LVX', 1)
    tyre['LXAL'] = readcheck(text_tyre, 'LXAL', 1)
    tyre['LCY'] = readcheck(text_tyre, 'LCY', 1)
    tyre['LMUY'] = readcheck(text_tyre, 'LMUY', 1)
    tyre['LEY'] = readcheck(text_tyre, 'LEY', 1)
    tyre['LKY'] = readcheck(text_tyre, 'LKY', 1)
    tyre['LKYC'] = readcheck(text_tyre, 'LKYC', 1)
    tyre['LKZC'] = readcheck(text_tyre, 'LKZC', 1)
    tyre['LHY'] = readcheck(text_tyre, 'LHY', 1)
    tyre['LVY'] = readcheck(text_tyre, 'LVY', 1)
    tyre['LYKA'] = readcheck(text_tyre, 'LYKA', 1)
    tyre['LVYKA'] = readcheck(text_tyre, 'LVYKA', 1)
    tyre['LMX'] = readcheck(text_tyre, 'LMX', 1)
    tyre['LVMX'] = readcheck(text_tyre, 'LVMX', 1)
    tyre['LMY'] = readcheck(text_tyre, 'LMY', 1)
    tyre['LTR'] = readcheck(text_tyre, 'LTR', 1)
    tyre['LRES'] = readcheck(text_tyre, 'LRES', 1)
    tyre['LS'] = readcheck(text_tyre, 'LS', 1)
    tyre['LMP'] = readcheck(text_tyre, 'LMP', 1)

    # Longitudinal coefficients
    tyre['PCX1'] = readcheck(text_tyre, 'PCX1', 1.65)
    tyre['PDX1'] = readcheck(text_tyre, 'PDX1', 1.3)
    tyre['PDX2'] = readcheck(text_tyre, 'PDX2', -0.15)
    tyre['PDX3'] = readcheck(text_tyre, 'PDX3', 0)
    tyre['PEX1'] = readcheck(text_tyre, 'PEX1', 0)
    tyre['PEX2'] = readcheck(text_tyre, 'PEX2', 0)
    tyre['PEX3'] = readcheck(text_tyre, 'PEX3', 0)
    tyre['PEX4'] = readcheck(text_tyre, 'PEX4', 0)
    tyre['PKX1'] = readcheck(text_tyre, 'PKX1', 20)
    tyre['PKX2'] = readcheck(text_tyre, 'PKX2', 0)
    tyre['PKX3'] = readcheck(text_tyre, 'PKX3', 0)
    tyre['PHX1'] = readcheck(text_tyre, 'PHX1', 0)
    tyre['PHX2'] = readcheck(text_tyre, 'PHX2', 0)
    tyre['PVX1'] = readcheck(text_tyre, 'PVX1', 0)
    tyre['PVX2'] = readcheck(text_tyre, 'PVX2', 0)
    tyre['RBX1'] = readcheck(text_tyre, 'RBX1', 0)
    tyre['RBX2'] = readcheck(text_tyre, 'RBX2', 0)
    tyre['RBX3'] = readcheck(text_tyre, 'RBX3', 0)
    tyre['RCX1'] = readcheck(text_tyre, 'RCX1', 0)
    tyre['REX1'] = readcheck(text_tyre, 'REX1', 0)
    tyre['REX2'] = readcheck(text_tyre, 'REX2', 0)
    tyre['RHX1'] = readcheck(text_tyre, 'RHX1', 0)
    tyre['PPX1'] = readcheck(text_tyre, 'PPX1', 0)
    tyre['PPX2'] = readcheck(text_tyre, 'PPX2', 0)
    tyre['PPX3'] = readcheck(text_tyre, 'PPX3', 0)
    tyre['PPX4'] = readcheck(text_tyre, 'PPX4', 0)

    # Lateral coefficients
    tyre['PCY1'] = readcheck(text_tyre, 'PCY1', 1.3)
    tyre['PDY1'] = readcheck(text_tyre, 'PDY1', 1.1)
    tyre['PDY2'] = readcheck(text_tyre, 'PDY2', -0.15)
    tyre['PDY3'] = readcheck(text_tyre, 'PDY3', 0)
    tyre['PEY1'] = readcheck(text_tyre, 'PEY1', 0)
    tyre['PEY2'] = readcheck(text_tyre, 'PEY2', 0)
    tyre['PEY3'] = readcheck(text_tyre, 'PEY3', 0)
    tyre['PEY4'] = readcheck(text_tyre, 'PEY4', 0)
    tyre['PEY5'] = readcheck(text_tyre, 'PEY5', 0)
    tyre['PKY1'] = readcheck(text_tyre, 'PKY1', -20)
    tyre['PKY2'] = readcheck(text_tyre, 'PKY2', 2)
    tyre['PKY3'] = readcheck(text_tyre, 'PKY3', 0)
    tyre['PKY4'] = readcheck(text_tyre, 'PKY4', 2)
    tyre['PKY5'] = readcheck(text_tyre, 'PKY5', 0)
    tyre['PKY6'] = readcheck(text_tyre, 'PKY6', -1)
    tyre['PKY7'] = readcheck(text_tyre, 'PKY7', 0)
    tyre['PHY1'] = readcheck(text_tyre, 'PHY1', 0)
    tyre['PHY2'] = readcheck(text_tyre, 'PHY2', 0)
    tyre['PVY1'] = readcheck(text_tyre, 'PVY1', 0)
    tyre['PVY2'] = readcheck(text_tyre, 'PVY2', 0)
    tyre['PVY3'] = readcheck(text_tyre, 'PVY3', 0)
    tyre['PVY4'] = readcheck(text_tyre, 'PVY4', 0)
    tyre['RBY1'] = readcheck(text_tyre, 'RBY1', 0)
    tyre['RBY2'] = readcheck(text_tyre, 'RBY2', 0)
    tyre['RBY3'] = readcheck(text_tyre, 'RBY3', 0)
    tyre['RBY4'] = readcheck(text_tyre, 'RBY4', 0)
    tyre['RCY1'] = readcheck(text_tyre, 'RCY1', 0)
    tyre['REY1'] = readcheck(text_tyre, 'REY1', 0)
    tyre['REY2'] = readcheck(text_tyre, 'REY2', 0)
    tyre['RHY1'] = readcheck(text_tyre, 'RHY1', 0)
    tyre['RHY2'] = readcheck(text_tyre, 'RHY2', 0)
    tyre['RVY1'] = readcheck(text_tyre, 'RVY1', 0)
    tyre['RVY2'] = readcheck(text_tyre, 'RVY2', 0)
    tyre['RVY3'] = readcheck(text_tyre, 'RVY3', 0)
    tyre['RVY4'] = readcheck(text_tyre, 'RVY4', 0)
    tyre['RVY5'] = readcheck(text_tyre, 'RVY5', 0)
    tyre['RVY6'] = readcheck(text_tyre, 'RVY6', 0)
    tyre['PPY1'] = readcheck(text_tyre, 'PPY1', 0)
    tyre['PPY2'] = readcheck(text_tyre, 'PPY2', 0)
    tyre['PPY3'] = readcheck(text_tyre, 'PPY3', 0)
    tyre['PPY4'] = readcheck(text_tyre, 'PPY4', 0)
    tyre['PPY5'] = readcheck(text_tyre, 'PPY5', 0)

    # Overturning coefficients
    tyre['QSX1'] = readcheck(text_tyre, 'QSX1', 0)
    tyre['QSX2'] = readcheck(text_tyre, 'QSX2', 0)
    tyre['QSX3'] = readcheck(text_tyre, 'QSX3', 0)
    tyre['QSX4'] = readcheck(text_tyre, 'QSX4', 0)
    tyre['QSX5'] = readcheck(text_tyre, 'QSX5', 0)
    tyre['QSX6'] = readcheck(text_tyre, 'QSX6', 0)
    tyre['QSX7'] = readcheck(text_tyre, 'QSX7', 0)
    tyre['QSX8'] = readcheck(text_tyre, 'QSX8', 0)
    tyre['QSX9'] = readcheck(text_tyre, 'QSX9', 0)
    tyre['QSX10'] = readcheck(text_tyre, 'QSX10', 0)
    tyre['QSX11'] = readcheck(text_tyre, 'QSX11', 0)
    tyre['QSX12'] = readcheck(text_tyre, 'QSX12', 0)
    tyre['QSX13'] = readcheck(text_tyre, 'QSX13', 0)
    tyre['QSX14'] = readcheck(text_tyre, 'QSX14', 0)
    tyre['PPMX1'] = readcheck(text_tyre, 'PPMX1', 0)

    # Rolling coefficients
    tyre['QSY1'] = readcheck(text_tyre, 'QSY1', 0.01)
    tyre['QSY2'] = readcheck(text_tyre, 'QSY2', 0)
    tyre['QSY3'] = readcheck(text_tyre, 'QSY3', 0)
    tyre['QSY4'] = readcheck(text_tyre, 'QSY4', 0)
    tyre['QSY5'] = readcheck(text_tyre, 'QSY5', 0)
    tyre['QSY6'] = readcheck(text_tyre, 'QSY6', 0)
    tyre['QSY7'] = readcheck(text_tyre, 'QSY7', 0.85)
    tyre['QSY8'] = readcheck(text_tyre, 'QSY8', 0)

    # Aligning coefficients
    tyre['QBZ1'] = readcheck(text_tyre, 'QBZ1', 10)
    tyre['QBZ2'] = readcheck(text_tyre, 'QBZ2', 0)
    tyre['QBZ3'] = readcheck(text_tyre, 'QBZ3', 0)
    tyre['QBZ4'] = readcheck(text_tyre, 'QBZ4', 0)
    tyre['QBZ5'] = readcheck(text_tyre, 'QBZ5', 0)
    tyre['QBZ9'] = readcheck(text_tyre, 'QBZ9', 30)
    tyre['QBZ10'] = readcheck(text_tyre, 'QBZ10', 0)
    tyre['QCZ1'] = readcheck(text_tyre, 'QCZ1', 1.1)
    tyre['QDZ1'] = readcheck(text_tyre, 'QDZ1', 0.12)
    tyre['QDZ2'] = readcheck(text_tyre, 'QDZ2', 0)
    tyre['QDZ3'] = readcheck(text_tyre, 'QDZ3', 0)
    tyre['QDZ4'] = readcheck(text_tyre, 'QDZ4', 0)
    tyre['QDZ6'] = readcheck(text_tyre, 'QDZ6', 0)
    tyre['QDZ7'] = readcheck(text_tyre, 'QDZ7', 0)
    tyre['QDZ8'] = readcheck(text_tyre, 'QDZ8', -0.1)
    tyre['QDZ9'] = readcheck(text_tyre, 'QDZ9', 0)
    tyre['QDZ10'] = readcheck(text_tyre, 'QDZ10', 0)
    tyre['QDZ11'] = readcheck(text_tyre, 'QDZ11', 0)
    tyre['QEZ1'] = readcheck(text_tyre, 'QEZ1', 0)
    tyre['QEZ2'] = readcheck(text_tyre, 'QEZ2', 0)
    tyre['QEZ3'] = readcheck(text_tyre, 'QEZ3', 0)
    tyre['QEZ4'] = readcheck(text_tyre, 'QEZ4', 0)
    tyre['QEZ5'] = readcheck(text_tyre, 'QEZ5', 0)
    tyre['QHZ1'] = readcheck(text_tyre, 'QHZ1', 0)
    tyre['QHZ2'] = readcheck(text_tyre, 'QHZ2', 0)
    tyre['QHZ3'] = readcheck(text_tyre, 'QHZ3', 0)
    tyre['QHZ4'] = readcheck(text_tyre, 'QHZ4', 0)
    tyre['SSZ1'] = readcheck(text_tyre, 'SSZ1', 0)
    tyre['SSZ2'] = readcheck(text_tyre, 'SSZ2', 0)
    tyre['SSZ3'] = readcheck(text_tyre, 'SSZ3', 0)
    tyre['SSZ4'] = readcheck(text_tyre, 'SSZ4', 0)
    tyre['PPZ1'] = readcheck(text_tyre, 'PPZ1', 0)
    tyre['PPZ2'] = readcheck(text_tyre, 'PPZ2', 0)

    # Additional parameters that might be needed
    tyre['LGYR'] = readcheck(text_tyre, 'LGYR', 1)
    tyre['LSGKP'] = readcheck(text_tyre, 'LSGKP', 1)
    tyre['LSGAL'] = readcheck(text_tyre, 'LSGAL', 1)

    return tyre