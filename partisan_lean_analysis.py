"""
Partisan Lean Analysis across Singapore General Elections (2011, 2020, 2025)
and US House Cook PVI & 2026 Midterm Elections.
Standardized scale comparison: PAP & Republicans on the LEFT, Opposition & Democrats on the RIGHT.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#EBEBEB'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

STATE_NAME_TO_ABBR = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

def parse_pvi_string(pvi_str):
    if pd.isnull(pvi_str) or not str(pvi_str).strip():
        return np.nan
    s = str(pvi_str).strip().upper()
    if s == 'EVEN':
        return 0.0
    if s.startswith('D+'):
        return float(s[2:])   # Positive = Democratic (Right)
    if s.startswith('R+'):
        return -float(s[2:])  # Negative = Republican (Left)
    return np.nan

def normalize_nyt_district_name(full_name):
    parts = full_name.split(' District ')
    if len(parts) == 2:
        st_name, dist_num = parts[0].strip(), parts[1].strip()
        st_abbr = STATE_NAME_TO_ABBR.get(st_name, st_name[:2].upper())
        return f"{st_abbr}-{int(dist_num):02d}"
    elif 'At-Large' in full_name:
        st_name = full_name.replace(' At-Large District', '').replace(' At-Large', '').strip()
        st_abbr = STATE_NAME_TO_ABBR.get(st_name, st_name[:2].upper())
        return f"{st_abbr}-AL"
    return full_name

def load_and_process_singapore_data():
    files = {
        2006: 'Singapore_2006_results.csv',
        2011: 'Singapore_2011_results.csv',
        2015: 'Singapore_2015_results.csv',
        2020: 'Singapore_2020_results.csv',
        2025: 'Singapore_2025_results.csv'
    }
    all_constituencies = []
    election_summaries = []
    
    for year, filepath in files.items():
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing file: {filepath}")
            
        df_raw = pd.read_csv(filepath)
        df_raw['Constituency'] = df_raw['Constituency'].str.strip().str.title()
        
        grouped = df_raw.groupby('Constituency')
        tot_pap_votes = 0
        tot_valid_votes = 0
        ward_records = []
        
        for const_name, group in grouped:
            is_uncontested = any(group['Result'].str.strip().str.lower() == 'uncontested')
            electors = group['Electors'].dropna().iloc[0] if len(group['Electors'].dropna()) > 0 else np.nan
            
            if is_uncontested:
                ward_records.append({
                    'Year': int(year),
                    'Constituency': const_name,
                    'Electors': electors,
                    'Contested': False,
                    'PAP_Votes': np.nan,
                    'Total_Votes': np.nan,
                    'PAP_Share_Pct': np.nan,
                    'Opp_Share_Pct': np.nan,
                    'Partisan_Coord_Pct': np.nan,  # Negative = PAP (Left), Positive = Opp (Right)
                    'PAP_Lean_Magnitude': np.nan,
                    'Partisan_Margin_Pct': np.nan
                })
                continue
                
            pap_rows = group[group['Party'].str.strip().str.upper() == 'PAP']
            if len(pap_rows) == 0:
                continue
                
            pap_votes = pd.to_numeric(pap_rows['Votes'].iloc[0], errors='coerce')
            tot_votes = pd.to_numeric(group['Votes'], errors='coerce').sum()
            
            if 'Vote_Share_Pct' in pap_rows and pd.notnull(pap_rows['Vote_Share_Pct'].iloc[0]):
                pap_share = float(pap_rows['Vote_Share_Pct'].iloc[0])
            else:
                pap_share = (pap_votes / tot_votes) * 100.0
                
            opp_share = 100.0 - pap_share
            # PAP on the LEFT (negative), Opposition on the RIGHT (positive)
            partisan_coord = 50.0 - pap_share
            pap_lean_magnitude = pap_share - 50.0
            partisan_margin = pap_share - opp_share
            
            tot_pap_votes += pap_votes
            tot_valid_votes += tot_votes
            
            ward_records.append({
                'Year': int(year),
                'Constituency': const_name,
                'Electors': electors,
                'Contested': True,
                'PAP_Votes': pap_votes,
                'Total_Votes': tot_votes,
                'PAP_Share_Pct': pap_share,
                'Opp_Share_Pct': opp_share,
                'Partisan_Coord_Pct': partisan_coord,
                'PAP_Lean_Magnitude': pap_lean_magnitude,
                'Partisan_Margin_Pct': partisan_margin
            })
            
        df_year = pd.DataFrame(ward_records)
        nat_pap_pct = (tot_pap_votes / tot_valid_votes) * 100.0
        nat_coord_pct = 50.0 - nat_pap_pct  # Negative (Left)
        nat_pap_lean = nat_pap_pct - 50.0
        nat_margin_pct = nat_pap_pct - (100.0 - nat_pap_pct)
        
        contested_only = df_year[df_year['Contested']]
        mean_coord_pct = contested_only['Partisan_Coord_Pct'].mean()
        median_coord_pct = contested_only['Partisan_Coord_Pct'].median()
        std_coord_pct = contested_only['Partisan_Coord_Pct'].std()
        
        all_constituencies.append(df_year)
        
        election_summaries.append({
            'Year': int(year),
            'Total_Wards': len(df_year),
            'Contested_Wards': len(contested_only),
            'Total_Valid_Votes': int(tot_valid_votes),
            'Total_PAP_Votes': int(tot_pap_votes),
            'National_PAP_Pct': nat_pap_pct,
            'National_Coord_Pct': nat_coord_pct,      # Negative (Left)
            'National_PAP_Lean_Pct': nat_pap_lean,    # Positive magnitude
            'National_Margin_Pct': nat_margin_pct,
            'Mean_Ward_Coord_Pct': mean_coord_pct,
            'Median_Ward_Coord_Pct': median_coord_pct,
            'Std_Ward_Coord_Pct': std_coord_pct,
            'Min_Ward_Coord_Pct': contested_only['Partisan_Coord_Pct'].min(), # Furthest Left (Strongest PAP)
            'Max_Ward_Coord_Pct': contested_only['Partisan_Coord_Pct'].max()  # Furthest Right (Strongest Opp)
        })
        
    return pd.concat(all_constituencies, ignore_index=True), pd.DataFrame(election_summaries)

def load_and_merge_us_data(cd_csv_path='congressional_districts_data.csv', nyt_json_path='2026 results-nyt.json'):
    df_cd = pd.read_csv(cd_csv_path)
    df_cd['PVI_2022_Numeric'] = df_cd['PVI_2022'].apply(parse_pvi_string)
    df_cd['PVI_2024_Numeric'] = df_cd['PVI_2024'].apply(parse_pvi_string)
    df_cd['PVI_2024_Coord'] = df_cd['PVI_2024_Numeric'] / 2.0  # Negative = R (Left), Positive = D (Right)
    
    with open(nyt_json_path, 'r', encoding='utf-8') as f:
        nyt_raw = json.load(f)
        
    seen = set()
    nyt_unique = []
    for item in nyt_raw:
        full = item.get('district_full')
        if full and full not in seen:
            seen.add(full)
            nyt_unique.append(item)
            
    nyt_records = []
    for item in nyt_unique:
        full = item.get('district_full')
        m_str = str(item.get('margin', '')).strip()
        cd_code = normalize_nyt_district_name(full)
        
        status = 'Contested'
        winner = 'TBD'
        margin = np.nan
        coord = np.nan
        
        if 'Uncontested' in m_str:
            status = 'Uncontested'
            winner = 'Democrat' if m_str.startswith('D') else 'Republican'
        elif m_str.startswith('D+'):
            margin = float(m_str[2:])
            coord = margin / 2.0   # Positive (Democrat on Right)
            winner = 'Democrat'
        elif m_str.startswith('R+'):
            margin = -float(m_str[2:])
            coord = margin / 2.0  # Negative (Republican on Left)
            winner = 'Republican'
        elif m_str in ['Even', '0']:
            margin = 0.0
            coord = 0.0
            winner = 'Even/Tied'
        else:
            status = m_str
            
        nyt_records.append({
            'District': cd_code,
            'District_Full': full,
            'Winner_2026': winner,
            'Margin_2026': margin,
            'Coord_2026': coord,
            'Status_2026': status
        })
        
    df_nyt = pd.DataFrame(nyt_records)
    df_us_merged = pd.merge(df_cd, df_nyt, on='District', how='left')
    df_us_merged['Swing_vs_PVI_2024'] = df_us_merged['Margin_2026'] - df_us_merged['PVI_2024_Numeric']
    
    contested = df_us_merged[df_us_merged['Status_2026'] == 'Contested']
    summary = {
        'Total_Districts': len(df_us_merged),
        'Contested_Districts': len(contested),
        'Uncontested_Districts': (df_us_merged['Status_2026'] == 'Uncontested').sum(),
        'Dem_Seats_2026': (df_us_merged['Winner_2026'] == 'Democrat').sum(),
        'Rep_Seats_2026': (df_us_merged['Winner_2026'] == 'Republican').sum(),
        'Mean_PVI_2024_Coord': df_us_merged['PVI_2024_Coord'].mean(),
        'Std_PVI_2024_Coord': df_us_merged['PVI_2024_Coord'].std(),
        'Mean_Dem_Coord_2026': contested['Coord_2026'].mean(),
        'Median_Dem_Coord_2026': contested['Coord_2026'].median(),
        'Std_Dem_Coord_2026': contested['Coord_2026'].std(),
        'Mean_Margin_2026': contested['Margin_2026'].mean(),
        'Mean_Swing_vs_PVI_2024': contested['Swing_vs_PVI_2024'].mean()
    }
    return df_us_merged, pd.DataFrame([summary])

TARGET_CONFIG = {
    'Aljunied': {'color': '#1f77b4', 'marker': 's', 'size': 110, 'label': 'Aljunied GRC'},
    'East Coast': {'color': '#ff7f0e', 'marker': 'D', 'size': 105, 'label': 'East Coast GRC'},
    'West Coast': {'color': '#2ca02c', 'marker': '^', 'size': 115, 'label': 'West Coast GRC'},
    'Tanjong Pagar': {'color': '#d62728', 'marker': '*', 'size': 160, 'label': 'Tanjong Pagar GRC'}
}

def match_target_ward(name):
    nl = str(name).lower().strip()
    if 'aljunied' in nl: return 'Aljunied'
    if 'east coast' in nl: return 'East Coast'
    if 'west coast' in nl: return 'West Coast'
    if 'tanjong pagar' in nl: return 'Tanjong Pagar'
    return None

def plot_faceted_distribution_same_scale(df_constituencies, df_summary, output_path='singapore_partisan_lean_same_scale.png'):
    SCALE_MIN = -35.0  # PAP on Left
    SCALE_MAX = 20.0   # Opp on Right
    years = sorted(df_summary['Year'].unique())
    palette = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a']
    colors = {str(yr): palette[idx % len(palette)] for idx, yr in enumerate(years)}
    
    fig, axes = plt.subplots(len(years), 1, figsize=(14, 3.1 * len(years)), sharex=True)
    fig.suptitle('Individual Constituency Partisan Leans Across Singapore General Elections (2006–2025)\n(Highlighting Tanjong Pagar, Aljunied, East Coast & West Coast; Non-Overlapping Labels; PAP on LEFT, Opp on RIGHT)', 
                 fontsize=14, fontweight='bold', y=0.99)
    
    df_constituencies['Target_Key'] = df_constituencies['Constituency'].apply(match_target_ward)

    for i, year in enumerate(years):
        ax = axes[i]
        year_data = df_constituencies[(df_constituencies['Year'] == year) & (df_constituencies['Contested'])].copy()
        sum_row = df_summary[df_summary['Year'] == year].iloc[0]
        
        nat_coord = sum_row['National_Coord_Pct']
        mean_coord = sum_row['Mean_Ward_Coord_Pct']
        color = colors[str(year)]
        
        ax.axvspan(SCALE_MIN, 0, color='#fee8e8', alpha=0.45, label='PAP Lean (Ruling Party)' if i == 0 else "")
        ax.axvspan(0, SCALE_MAX, color='#e8f0fe', alpha=0.45, label='Opposition Lean' if i == 0 else "")
        ax.axvline(0, color='#444444', linestyle=':', linewidth=1.4, alpha=0.8, label='50-50 Parity' if i == 0 else "")
        
        sns.kdeplot(year_data['Partisan_Coord_Pct'], ax=ax, color=color, fill=True, alpha=0.20, linewidth=2.0, bw_adjust=0.8)
        
        # Plot ALL other constituencies as standard dots
        np.random.seed(42 + year)
        jitter = np.random.normal(0, 0.0025, size=len(year_data))
        year_data['Jitter_Y'] = jitter + 0.012
        
        non_target = year_data[year_data['Target_Key'].isnull()]
        ax.scatter(non_target['Partisan_Coord_Pct'], non_target['Jitter_Y'], 
                   color=color, edgecolor='black', linewidth=0.6, s=55, alpha=0.65, zorder=4,
                   label=f'Other Contested Wards (n={len(non_target)})')
        
        # Walkovers in this year
        if year == 2006:
            ax.text(-34.2, 0.051, 'Walkovers (100% PAP Seats):\n★ Tanjong Pagar (6 MPs) | ▲ West Coast (5 MPs)', 
                    fontsize=8.5, fontweight='bold', color='#8b0000',
                    bbox=dict(boxstyle='round,pad=0.35', facecolor='#fff5f5', edgecolor='#d62728', alpha=0.95))
        elif year == 2011:
            ax.text(-34.2, 0.051, 'Walkover (100% PAP Seats):\n★ Tanjong Pagar (5 MPs)', 
                    fontsize=8.5, fontweight='bold', color='#8b0000',
                    bbox=dict(boxstyle='round,pad=0.35', facecolor='#fff5f5', edgecolor='#d62728', alpha=0.95))
            
        # Target Wards: High-priority highlights with custom non-overlapping staggered anchors
        target_rows = year_data[year_data['Target_Key'].notnull()]
        for _, r in target_rows.iterrows():
            t_key = r['Target_Key']
            cfg = TARGET_CONFIG[t_key]
            cx = r['Partisan_Coord_Pct']
            cy = r['Jitter_Y']
            
            # Scatter highlighted point
            ax.scatter(cx, cy, color=cfg['color'], marker=cfg['marker'], s=cfg['size'], 
                       edgecolor='black', linewidth=1.2, zorder=7)
            
            # Base label text
            if r['PAP_Share_Pct'] >= 50.0:
                txt = f"★ {r['Constituency']}\n(PAP {r['PAP_Share_Pct']:.1f}%)"
            else:
                txt = f"★ {r['Constituency']}\n(Opp {r['Opp_Share_Pct']:.1f}%)"
                
            # Stagger logic per year to prevent ANY overlapping
            if year == 2006:
                if t_key == 'East Coast':
                    tx, ty, ha = cx - 0.5, 0.038, 'right'
                else: # Aljunied
                    tx, ty, ha = cx + 0.5, 0.038, 'left'
            elif year == 2011:
                if t_key == 'West Coast':
                    tx, ty, ha = cx, 0.038, 'center'
                elif t_key == 'East Coast':
                    tx, ty, ha = cx - 0.5, 0.040, 'right'
                else: # Aljunied
                    tx, ty, ha = cx + 0.5, 0.038, 'left'
            elif year == 2015:
                if t_key == 'West Coast':
                    tx, ty, ha = cx - 2.5, 0.026, 'right'
                elif t_key == 'Tanjong Pagar':
                    tx, ty, ha = cx + 2.5, 0.048, 'left'
                elif t_key == 'East Coast':
                    tx, ty, ha = cx - 0.5, 0.038, 'right'
                else: # Aljunied
                    tx, ty, ha = cx + 0.5, 0.038, 'left'
            elif year == 2020:
                if t_key == 'Tanjong Pagar':
                    tx, ty, ha = cx, 0.038, 'center'
                elif t_key == 'East Coast':
                    tx, ty, ha = cx - 2.8, 0.046, 'right'
                elif t_key == 'West Coast':
                    tx, ty, ha = cx + 2.8, 0.026, 'left'
                else: # Aljunied
                    tx, ty, ha = cx + 0.5, 0.038, 'left'
            elif year == 2025:
                if t_key == 'Tanjong Pagar':
                    tx, ty, ha = cx, 0.038, 'center'
                elif t_key == 'West Coast':
                    tx, ty, ha = cx - 3.2, 0.026, 'right'
                elif t_key == 'East Coast':
                    tx, ty, ha = cx + 3.2, 0.048, 'left'
                else: # Aljunied
                    tx, ty, ha = cx + 0.5, 0.038, 'left'
                    
            ax.annotate(txt, xy=(cx, cy), xytext=(tx, ty),
                        fontsize=8.0, fontweight='bold', color=cfg['color'], ha=ha, va='center',
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=cfg['color'], alpha=0.92, lw=1.1),
                        arrowprops=dict(arrowstyle='->', color=cfg['color'], lw=1.2), zorder=8)
            
        # Average and Mean lines
        ax.axvline(nat_coord, color='#b30000', linestyle='--', linewidth=2.3, zorder=5,
                   label=f'Avg PAP Lean: +{abs(nat_coord):.2f}%')
        ax.axvline(mean_coord, color='#e66101', linestyle='-.', linewidth=1.6, alpha=0.85, zorder=5,
                   label=f'Mean Ward: {mean_coord:.2f}%')
        
        ax.set_ylabel(f'{year} Density', fontsize=11, fontweight='bold')
        ax.set_xlim(SCALE_MIN, SCALE_MAX)
        ax.set_ylim(-0.005, 0.068)
        ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=7.8)
        
    axes[-1].set_xlabel('Partisan Lean Axis (% relative to 50% Competitive Parity)\n[<-- PAP Leaning (Ruling Party) | 50-50 Parity | Opposition Leaning -->]', 
                       fontsize=12, fontweight='bold', labelpad=10)
    ticks = np.arange(SCALE_MIN, SCALE_MAX + 1, 10)
    def format_tick(t):
        if t < 0: return f"PAP +{abs(t):.0f}%\n({50-t:.0f}% PAP)"
        elif t > 0: return f"Opp +{t:.0f}%\n({50-t:.0f}% PAP)"
        return "0%\n(Parity)"
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([format_tick(t) for t in ticks], fontsize=9.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_path} (PAP on Left, Target Wards Highlighted)")

def plot_trend_and_overlay(df_constituencies, df_summary, output_path='singapore_partisan_lean_trend_and_overlay.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))
    fig.suptitle('Longitudinal Trajectories of Singapore General Elections (2006–2025)\n(Prominently Highlighting Tanjong Pagar, Aljunied, East Coast & West Coast with Anti-Collision Labels)', 
                 fontsize=14, fontweight='bold', y=0.98)

    years_arr = df_summary['Year'].values
    nat_coords = df_summary['National_Coord_Pct'].values
    mean_coords = df_summary['Mean_Ward_Coord_Pct'].values

    # Panel 1: Trajectories
    ax1.axhline(0, color='gray', linestyle=':', linewidth=1.5, label='50-50 Parity Baseline')
    ax1.axhspan(-36, 0, color='#fee8e8', alpha=0.30, label='PAP Advantage Zone (Left / Negative)')
    ax1.axhspan(0, 20, color='#e8f0fe', alpha=0.30, label='Opposition Advantage Zone (Right / Positive)')

    # Background dots for ALL other individual constituencies
    np.random.seed(42)
    for yr in years_arr:
        yr_data = df_constituencies[(df_constituencies['Year'] == yr) & (df_constituencies['Contested'])]
        other_wards = yr_data[yr_data['Target_Key'].isnull()]
        jit = np.random.uniform(-0.15, 0.15, size=len(other_wards))
        ax1.scatter(yr + jit, other_wards['Partisan_Coord_Pct'], color='#999999', edgecolor='white', 
                    s=45, alpha=0.45, zorder=3, label='Other Contested Constituencies' if yr == 2006 else "")

    # Overlay National Weighted Average
    ax1.plot(years_arr, nat_coords, marker='o', markersize=8, color='#800000', linewidth=2.8, 
             linestyle='-', label='National Weighted Avg (PAP on Left)', zorder=6)

    # Highlight Target Wards
    traj_data = {
        'Aljunied': {'years': [2006, 2011, 2015, 2020, 2025], 'coords': [-6.09, 4.72, 0.96, 9.95, 9.71]},
        'East Coast': {'years': [2006, 2011, 2015, 2020, 2025], 'coords': [-13.86, -4.83, -10.73, -3.39, -8.73]},
        'West Coast': {'years': [2011, 2015, 2020, 2025], 'coords': [-16.57, -28.57, -1.68, -9.99], 'walkovers': [2006]},
        'Tanjong Pagar': {'years': [2015, 2020, 2025], 'coords': [-27.71, -13.10, -31.02], 'walkovers': [2006, 2011]}
    }

    for t_key, d in traj_data.items():
        cfg = TARGET_CONFIG[t_key]
        xs = d['years']
        ys = d['coords']
        ax1.plot(xs, ys, color=cfg['color'], linewidth=2.5, marker=cfg['marker'], 
                 markersize=cfg['size']**0.5, markeredgecolor='black', markeredgewidth=1.1,
                 label=f"★ {cfg['label']}", zorder=7)
        
        # Walkover indicators
        if 'walkovers' in d:
            for w_yr in d['walkovers']:
                ax1.scatter([w_yr], [-34.0], marker=cfg['marker'], s=cfg['size']*0.8, 
                            facecolor='none', edgecolor=cfg['color'], linewidth=1.8, zorder=7)
                # Dotted line to first contested
                ax1.plot([w_yr, xs[0]], [-34.0, ys[0]], color=cfg['color'], linestyle=':', linewidth=1.3, alpha=0.7)

    # Annotations at 2025 Terminus (Carefully staggered to avoid overlap)
    ax1.annotate('★ Aljunied GRC (WP 59.7%)\n[Held by WP since 2011]', 
                 xy=(2025, 9.71), xytext=(2025.3, 10.5),
                 fontsize=8.5, fontweight='bold', color=TARGET_CONFIG['Aljunied']['color'], va='center',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=TARGET_CONFIG['Aljunied']['color'], alpha=0.95),
                 arrowprops=dict(arrowstyle='->', color=TARGET_CONFIG['Aljunied']['color'], lw=1.2), zorder=8)

    ax1.annotate('★ East Coast GRC (PAP 58.7%)\n[Persistent 50-55% Battleground]', 
                 xy=(2025, -8.73), xytext=(2025.3, -5.5),
                 fontsize=8.5, fontweight='bold', color=TARGET_CONFIG['East Coast']['color'], va='center',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=TARGET_CONFIG['East Coast']['color'], alpha=0.95),
                 arrowprops=dict(arrowstyle='->', color=TARGET_CONFIG['East Coast']['color'], lw=1.2), zorder=8)

    ax1.annotate('★ West Coast-Jurong W (PAP 60.0%)\n[PSP 48.3% Close-Call in 2020]', 
                 xy=(2025, -9.99), xytext=(2025.3, -12.5),
                 fontsize=8.5, fontweight='bold', color=TARGET_CONFIG['West Coast']['color'], va='center',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=TARGET_CONFIG['West Coast']['color'], alpha=0.95),
                 arrowprops=dict(arrowstyle='->', color=TARGET_CONFIG['West Coast']['color'], lw=1.2), zorder=8)

    ax1.annotate('★ Tanjong Pagar GRC (PAP 81.0%)\n[LKY Walkover Fortress to 81% PAP]', 
                 xy=(2025, -31.02), xytext=(2025.3, -31.02),
                 fontsize=8.5, fontweight='bold', color=TARGET_CONFIG['Tanjong Pagar']['color'], va='center',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=TARGET_CONFIG['Tanjong Pagar']['color'], alpha=0.95),
                 arrowprops=dict(arrowstyle='->', color=TARGET_CONFIG['Tanjong Pagar']['color'], lw=1.2), zorder=8)

    ax1.annotate('National Weighted Avg\n(PAP +15.8% / 65.8%)', 
                 xy=(2025, nat_coords[-1]), xytext=(2025.3, -18.5),
                 fontsize=8.5, fontweight='bold', color='#800000', va='center',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='#fff5f5', edgecolor='#800000', alpha=0.95),
                 arrowprops=dict(arrowstyle='->', color='#800000', lw=1.2), zorder=8)

    # Landmark historic callouts (Zero overlap)
    ax1.annotate('2011: Historic WP Flip\n(PAP 45.3% vs WP 54.7%)', 
                 xy=(2011, 4.72), xytext=(2009.5, 14.0),
                 fontsize=8.2, fontweight='bold', color=TARGET_CONFIG['Aljunied']['color'],
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=TARGET_CONFIG['Aljunied']['color'], alpha=0.95),
                 arrowprops=dict(arrowstyle='->', color=TARGET_CONFIG['Aljunied']['color'], lw=1.2), zorder=8)

    ax1.annotate('2020: Razor-Thin PAP Win\n(PAP 51.7% vs PSP 48.3%)', 
                 xy=(2020, -1.68), xytext=(2017.5, 2.5),
                 fontsize=8.2, fontweight='bold', color=TARGET_CONFIG['West Coast']['color'],
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=TARGET_CONFIG['West Coast']['color'], alpha=0.95),
                 arrowprops=dict(arrowstyle='->', color=TARGET_CONFIG['West Coast']['color'], lw=1.2), zorder=8)

    ax1.text(2006, -35.2, 'Walkovers\n(LKY / PAP 100%)', fontsize=7.8, ha='center', color='#555555', style='italic')
    ax1.text(2011, -35.2, 'Walkover\n(LKY / PAP 100%)', fontsize=7.8, ha='center', color='#555555', style='italic')

    ax1.set_title('Longitudinal Trajectories of 4 Anchor Constituencies (2006–2025)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('General Election Year', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Partisan Coordinate (% relative to Parity)\n[<-- PAP Advantage (Left) | Opposition Advantage (Right) -->]', fontsize=10.5, fontweight='bold')
    ax1.set_xticks(years_arr)
    ax1.set_xticklabels([f'{yr} GE' for yr in years_arr], fontsize=10)
    ax1.set_xlim(2005.5, 2028.5)
    ax1.set_ylim(-37, 18)
    ax1.legend(loc='lower left', frameon=True, fontsize=8.0)

    # Panel 2: Density Overlay with Strip Highlights
    SCALE_MIN = -35.0
    SCALE_MAX = 20.0
    palette = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a']
    colors = {str(yr): palette[idx % len(palette)] for idx, yr in enumerate(years_arr)}
    
    ax2.axvline(0, color='gray', linestyle=':', linewidth=1.5, label='50-50 Parity')
    ax2.axvspan(SCALE_MIN, 0, color='#fee8e8', alpha=0.25)
    ax2.axvspan(0, SCALE_MAX, color='#e8f0fe', alpha=0.25)

    y_strip_offset = 0.0055
    for idx, yr in enumerate(years_arr):
        yr_data = df_constituencies[(df_constituencies['Year'] == yr) & (df_constituencies['Contested'])]
        c = colors[str(yr)]
        
        sns.kdeplot(yr_data['Partisan_Coord_Pct'], ax=ax2, color=c, linewidth=2.0, 
                    label=f'{yr} (PAP +{abs(df_summary[df_summary["Year"]==yr]["National_Coord_Pct"].iloc[0]):.1f}%)',
                    bw_adjust=0.85)
        
        # Scatter non-target dots
        non_t = yr_data[yr_data['Target_Key'].isnull()]
        ax2.scatter(non_t['Partisan_Coord_Pct'], [y_strip_offset * (idx + 1)] * len(non_t), 
                    color=c, edgecolor='black', linewidth=0.5, s=40, alpha=0.6, zorder=4)
        
        # Scatter highlighted target dots on strip
        t_rows = yr_data[yr_data['Target_Key'].notnull()]
        for _, r in t_rows.iterrows():
            cfg = TARGET_CONFIG[r['Target_Key']]
            ax2.scatter(r['Partisan_Coord_Pct'], y_strip_offset * (idx + 1), 
                        color=cfg['color'], marker=cfg['marker'], s=cfg['size']*0.7, 
                        edgecolor='black', linewidth=1.1, zorder=6)

    ax2.set_title('Overlay of Constituency Distributions with Strip Scatter\n(Highlighted Symbols Mark Aljunied, East Coast, West Coast, Tanjong Pagar)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Partisan Lean Axis (%)\n[<-- PAP Leaning (Left) | Opposition Leaning (Right) -->]', fontsize=10.5, fontweight='bold')
    ax2.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
    ax2.set_xlim(SCALE_MIN, SCALE_MAX)
    
    ticks = np.arange(SCALE_MIN, SCALE_MAX + 1, 10)
    def format_tick(t):
        if t < 0: return f"PAP +{abs(t):.0f}%\n({50-t:.0f}% PAP)"
        elif t > 0: return f"Opp +{t:.0f}%\n({50-t:.0f}% PAP)"
        return "0%\n(Parity)"
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([format_tick(t) for t in ticks], fontsize=8.5)
    ax2.legend(loc='upper right', frameon=True, fontsize=8.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_path} (PAP on Left, Target Wards Highlighted)")

def plot_three_panel_comparison(df_constituencies, df_summary, df_us_merged, df_us_summary, output_path='singapore_vs_us_pvi_and_2026.png'):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Cross-System Partisan Lean: Singapore 2025 vs US PVI Baseline vs US 2026 Midterms\n(Standardized Bipolar Scale: PAP & Republicans on LEFT, Opposition & Democrats on RIGHT)', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    SHARED_MIN = -40.0
    SHARED_MAX = 45.0
    
    for ax in [ax1, ax2, ax3]:
        ax.axvspan(SHARED_MIN, 0, color='#fee8e8', alpha=0.45)
        ax.axvspan(0, SHARED_MAX, color='#e8f0fe', alpha=0.45)
        ax.axvline(0, color='gray', linestyle=':', linewidth=1.4)
        ax.set_xlim(SHARED_MIN, SHARED_MAX)
        
    # Singapore 2025 (PAP on Left)
    sg_2025 = df_constituencies[(df_constituencies['Year'] == 2025) & (df_constituencies['Contested'])]
    sg_coord = df_summary[df_summary['Year'] == 2025]['National_Coord_Pct'].iloc[0]
    sns.kdeplot(sg_2025['Partisan_Coord_Pct'], ax=ax1, color='#7570b3', fill=True, alpha=0.3, linewidth=2.4, bw_adjust=0.85)
    np.random.seed(42)
    ax1.scatter(sg_2025['Partisan_Coord_Pct'], np.random.normal(0, 0.002, len(sg_2025)) + 0.012, 
                color='#7570b3', edgecolor='white', s=60, alpha=0.9, zorder=4, label=f'Singapore 2025 Wards (n={len(sg_2025)})')
    ax1.axvline(sg_coord, color='#b30000', linestyle='--', linewidth=2.8, zorder=5, label=f'Avg PAP Lean: +{abs(sg_coord):.2f}% (PAP 65.57%)')
    ax1.annotate(f"Singapore 2025 Avg: PAP +{abs(sg_coord):.2f}%\n(Coord: {sg_coord:.2f}% on Left)", 
                 xy=(sg_coord, 0.038), xytext=(sg_coord - 12.0, 0.035),
                 arrowprops=dict(facecolor='#b30000', shrink=0.08, width=1.5, headwidth=6),
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#b30000', alpha=0.95),
                 fontsize=9.5, fontweight='bold', color='#800000')
    ax1.set_title('Singapore 2025 General Election: Unimodal Ruling-Party Hegemony (Concentrated on LEFT)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('SG Density', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, fontsize=9)
    
    # US Cook PVI 2024 (GOP on Left, Dem on Right)
    pvi_valid = df_us_merged.dropna(subset=['PVI_2024_Coord'])
    pvi_avg_coord = df_us_summary['Mean_PVI_2024_Coord'].iloc[0]
    sns.kdeplot(pvi_valid['PVI_2024_Coord'], ax=ax2, color='#4575b4', fill=True, alpha=0.3, linewidth=2.4, bw_adjust=0.75)
    ax2.scatter(pvi_valid['PVI_2024_Coord'], np.random.normal(0, 0.002, len(pvi_valid)) + 0.012, 
                color='#4575b4', edgecolor='white', s=30, alpha=0.6, zorder=4, label=f'US Districts PVI 2024 (n={len(pvi_valid)})')
    ax2.axvline(pvi_avg_coord, color='#b30000', linestyle='--', linewidth=2.8, zorder=5, label=f'PVI Baseline Mean: +{pvi_avg_coord:.2f}% (Near Parity)')
    ax2.annotate(f"US Cook PVI 2024 Baseline: +{pvi_avg_coord:.2f}%\n(Symmetrical Bimodal Polarization)", 
                 xy=(pvi_avg_coord, 0.04), xytext=(pvi_avg_coord + 4.0, 0.038),
                 arrowprops=dict(facecolor='#b30000', shrink=0.08, width=1.5, headwidth=6),
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#b30000', alpha=0.95),
                 fontsize=9.5, fontweight='bold', color='#800000')
    ax2.set_title('US House Structural Baseline: Cook PVI 2024 (Republicans on LEFT, Democrats on RIGHT)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('PVI Density', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, fontsize=9)
    
    # US 2026 Actual (GOP on Left, Dem on Right)
    us_2026 = df_us_merged.dropna(subset=['Coord_2026'])
    us_avg_coord = df_us_summary['Mean_Dem_Coord_2026'].iloc[0]
    sns.kdeplot(us_2026['Coord_2026'], ax=ax3, color='#2b5c8f', fill=True, alpha=0.3, linewidth=2.4, bw_adjust=0.75)
    ax3.scatter(us_2026['Coord_2026'], np.random.normal(0, 0.002, len(us_2026)) + 0.008, 
                color='#2b5c8f', edgecolor='white', s=30, alpha=0.6, zorder=4, label=f'US 2026 Midterm Districts (n={len(us_2026)})')
    ax3.axvline(us_avg_coord, color='#b30000', linestyle='--', linewidth=2.8, zorder=5, label=f'2026 Mean: D+{us_avg_coord:.2f}% (Right)')
    ax3.annotate(f"US 2026 Midterm Mean: D+{us_avg_coord:.2f}%\n(Democratic Wave Swing to the Right)", 
                 xy=(us_avg_coord, 0.024), xytext=(us_avg_coord + 4.0, 0.022),
                 arrowprops=dict(facecolor='#b30000', shrink=0.08, width=1.5, headwidth=6),
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#b30000', alpha=0.95),
                 fontsize=9.5, fontweight='bold', color='#800000')
    ax3.set_title('US House 2026 Midterm Results: Democratic Wave Shift across Polarized Landscape', fontsize=12, fontweight='bold')
    ax3.set_ylabel('2026 Density', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Standardized Partisan Lean Scale (% relative to 50% Competitive Parity)\n[<-- PAP / Republican Lean (Red) | 50-50 Parity | Opposition / Democratic Lean (Blue) -->]', 
                   fontsize=12, fontweight='bold', labelpad=10)
    ax3.legend(loc='upper right', frameon=True, fontsize=9)
    
    ticks = np.arange(SHARED_MIN, SHARED_MAX + 1, 10)
    def format_shared_tick(t):
        if t < 0: return f"PAP / R+{abs(t):.0f}%\n({50-t:.0f}% PAP)"
        elif t > 0: return f"Opp / D+{t:.0f}%\n({50-t:.0f}% PAP)"
        return "0%\n(Parity)"
    ax3.set_xticks(ticks)
    ax3.set_xticklabels([format_shared_tick(t) for t in ticks], fontsize=8.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_path} (PAP & Republicans aligned on LEFT)")

def compute_dummymandering_metrics(df_us_merged):
    """
    Computes cracking zones, Mean-Median difference, and threshold swings to break
    2/3 constitutional supermajorities and 50% parliamentary majorities.
    """
    import math
    sg_elections = {}
    elections_info = [
        (2006, 'Singapore_2006_results.csv'),
        (2011, 'Singapore_2011_results.csv'),
        (2015, 'Singapore_2015_results.csv'),
        (2020, 'Singapore_2020_results.csv'),
        (2025, 'Singapore_2025_results.csv')
    ]
    for yr, fpath in elections_info:
        if not os.path.exists(fpath):
            continue
        df_raw = pd.read_csv(fpath)
        df_raw['Constituency'] = df_raw['Constituency'].str.strip().str.title()
        grouped = df_raw.groupby('Constituency')
        
        ward_data = []
        total_mps = 0
        uncontested_mps = 0
        
        for c_name, group in grouped:
            is_unc = any(group['Result'].str.strip().str.lower() == 'uncontested')
            pap_rows = group[group['Party'].str.strip().str.upper() == 'PAP']
            if len(pap_rows) == 0: continue
            
            cands = [x.strip() for x in str(pap_rows['Candidates'].iloc[0]).split(';') if x.strip()]
            mps = len(cands)
            total_mps += mps
            
            if is_unc:
                uncontested_mps += mps
            else:
                pap_share = float(pap_rows['Vote_Share_Pct'].iloc[0]) if pd.notnull(pap_rows['Vote_Share_Pct'].iloc[0]) else 100.0
                ward_data.append({
                    'Constituency': c_name,
                    'MPs': mps,
                    'PAP_Share': pap_share,
                    'Opp_Share': 100.0 - pap_share,
                    'PAP_Margin': pap_share - (100.0 - pap_share),
                    'Swing_Needed_To_Flip': max(0.0, pap_share - 50.0),
                    'Won_By_PAP': pap_share > 50.0
                })
                
        df_wards = pd.DataFrame(ward_data).sort_values(by='PAP_Share', ascending=True).reset_index(drop=True)
        
        mean_pap = df_wards['PAP_Share'].mean()
        median_pap = df_wards['PAP_Share'].median()
        mean_med_diff = mean_pap - median_pap
        
        # Volatile Cracking Zone (50.0% to 55.0% PAP vote share)
        cracking_zone = df_wards[(df_wards['PAP_Share'] >= 50.0) & (df_wards['PAP_Share'] <= 55.0)]
        cracked_wards_count = len(cracking_zone)
        cracked_mps_count = cracking_zone['MPs'].sum()
        
        # Thresholds in MPs
        supermaj_thresh_mps = math.ceil(total_mps * 2.0 / 3.0)
        opp_break_23_mps = total_mps - supermaj_thresh_mps + 1
        maj_thresh_mps = math.ceil(total_mps / 2.0)
        opp_break_50_mps = total_mps - maj_thresh_mps + 1
        
        cum_opp_mps = 0
        tipping_ward_23 = None
        swing_23 = None
        tipping_ward_50 = None
        swing_50 = None
        
        for idx, row in df_wards.iterrows():
            cum_opp_mps += row['MPs']
            if cum_opp_mps >= opp_break_23_mps and tipping_ward_23 is None:
                tipping_ward_23 = row['Constituency']
                swing_23 = row['Swing_Needed_To_Flip']
            if cum_opp_mps >= opp_break_50_mps and tipping_ward_50 is None:
                tipping_ward_50 = row['Constituency']
                swing_50 = row['Swing_Needed_To_Flip']
                
        sg_elections[yr] = {
            'Year': yr,
            'Total_MPs': total_mps,
            'Uncontested_MPs': uncontested_mps,
            'Mean_PAP': mean_pap,
            'Median_PAP': median_pap,
            'Mean_Median_Diff': mean_med_diff,
            'Cracked_Wards': cracked_wards_count,
            'Cracked_MPs': cracked_mps_count,
            'Cracked_MPs_Pct': (cracked_mps_count / total_mps) * 100.0,
            'Opp_MPs_Break_23': opp_break_23_mps,
            'Tipping_Ward_23': tipping_ward_23,
            'Swing_Break_23': swing_23,
            'Vote_Share_Break_23': mean_pap - swing_23,
            'Opp_MPs_Break_50': opp_break_50_mps,
            'Tipping_Ward_50': tipping_ward_50,
            'Swing_Break_50': swing_50,
            'Vote_Share_Break_50': mean_pap - swing_50,
            'Ward_DF': df_wards
        }
        
    us_contested = df_us_merged[df_us_merged['Status_2026'] == 'Contested'].copy()
    us_dem_shares = (us_contested['Margin_2026'] + 100.0) / 2.0
    us_mean_dem = us_dem_shares.mean()
    us_median_dem = us_dem_shares.median()
    us_mean_med_diff = us_mean_dem - us_median_dem
    
    us_cracked = us_contested[us_contested['Margin_2026'].abs() <= 10.0]
    us_cracked_seats = len(us_cracked)
    
    sorted_margins = us_contested['Margin_2026'].sort_values().values
    tipping_margin_us = sorted_margins[len(sorted_margins) - 218]
    swing_us_majority = tipping_margin_us / 2.0
    dem_seats_count = int((df_us_merged['Winner_2026'] == 'Democrat').sum())
    
    us_metrics = {
        'Total_Seats': 435,
        'Mean_Dem': us_mean_dem,
        'Median_Dem': us_median_dem,
        'Mean_Median_Diff': us_mean_med_diff,
        'Cracked_Seats': us_cracked_seats,
        'Cracked_Seats_Pct': (us_cracked_seats / 435.0) * 100.0,
        'Swing_Flip_Majority': swing_us_majority,
        'Vote_Share_Flip_Majority': us_mean_dem - swing_us_majority,
        'Dem_Seats_2026': dem_seats_count
    }
    
    return sg_elections, us_metrics

def plot_seats_votes_curves(sg_metrics, us_metrics, df_us_merged, output_path='dummymandering_seats_votes_curves.png'):
    """
    Plots the Seats-Votes Curves for Singapore (2020 vs 2025) and US House 2026,
    annotating the 2/3 Supermajority and 50% Majority thresholds.
    """
    fig, (ax_sg, ax_us) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('The Dummymander Effect & Seats-Votes Curves: Popular Vote vs Seats Won\n(Quantifying Key Partisan Swings Required to Break 2/3 Supermajorities and 50% Majorities)', 
                 fontsize=14, fontweight='bold', y=0.98)

    sim_votes = np.linspace(35, 80, 250)

    def simulate_sg_curve(yr):
        meta = sg_metrics[yr]
        df_w = meta['Ward_DF']
        tot_mps = meta['Total_MPs']
        unc_mps = meta['Uncontested_MPs']
        mean_p = meta['Mean_PAP']
        
        seat_pcts = []
        for target_v in sim_votes:
            swing = target_v - mean_p
            won_mps = unc_mps + sum(row['MPs'] for _, row in df_w.iterrows() if (row['PAP_Share'] + swing) > 50.0)
            seat_pcts.append((won_mps / tot_mps) * 100.0)
        return np.array(seat_pcts)

    palette_sg = {
        2006: ('#1f78b4', ':', '2006'),
        2011: ('#33a02c', '--', '2011'),
        2015: ('#e31a1c', '-.', '2015'),
        2020: ('#d95f02', '-', '2020'),
        2025: ('#7570b3', '-', '2025')
    }
    for yr in sorted(sg_metrics.keys()):
        c, ls, lbl = palette_sg.get(yr, ('#888888', '-', str(yr)))
        s_curve = simulate_sg_curve(yr)
        lw = 2.8 if yr == 2025 else (2.4 if yr == 2020 else 1.8)
        ax_sg.plot(sim_votes, s_curve, color=c, linestyle=ls, linewidth=lw, label=f'Singapore {yr} Seats-Votes Curve')

    ax_sg.axhline(66.67, color='#b30000', linestyle='--', linewidth=2.0, zorder=5)
    ax_sg.text(36, 67.5, '2/3 Supermajority Threshold (66.7% / 65 MPs in 2025)', color='#b30000', fontweight='bold', fontsize=9.5)

    ax_sg.axhline(50.0, color='#08519c', linestyle='-.', linewidth=2.0, zorder=5)
    ax_sg.text(36, 51.0, '50% Simple Majority Threshold (50.0% / 49 MPs in 2025)', color='#08519c', fontweight='bold', fontsize=9.5)

    v_23_2025 = sg_metrics[2025]['Vote_Share_Break_23']
    v_50_2025 = sg_metrics[2025]['Vote_Share_Break_50']
    ax_sg.axvline(v_23_2025, color='#b30000', linestyle=':', linewidth=1.8, alpha=0.9)
    ax_sg.axvline(v_50_2025, color='#08519c', linestyle=':', linewidth=1.8, alpha=0.9)

    ax_sg.annotate(f'2/3 Broken at {v_23_2025:.1f}% PAP Vote\n(Swing needed: +{sg_metrics[2025]["Swing_Break_23"]:.1f}%)',
                   xy=(v_23_2025, 66.67), xytext=(v_23_2025 - 11, 74),
                   arrowprops=dict(facecolor='#b30000', shrink=0.08, width=1.5, headwidth=6),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff5f5', edgecolor='#b30000', alpha=0.95),
                   fontsize=9, fontweight='bold', color='#800000')

    ax_sg.annotate(f'50% Broken at {v_50_2025:.1f}% PAP Vote\n(Swing needed: +{sg_metrics[2025]["Swing_Break_50"]:.1f}%)',
                   xy=(v_50_2025, 50.0), xytext=(v_50_2025 - 11, 40),
                   arrowprops=dict(facecolor='#08519c', shrink=0.08, width=1.5, headwidth=6),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#eff6ff', edgecolor='#08519c', alpha=0.95),
                   fontsize=9, fontweight='bold', color='#08519c')

    ax_sg.scatter([sg_metrics[2025]['Mean_PAP']], [93.8], color='#7570b3', s=120, edgecolor='black', zorder=6)
    ax_sg.text(sg_metrics[2025]['Mean_PAP'] + 1, 91.5, f"2025 Actual:\n{sg_metrics[2025]['Mean_PAP']:.1f}% Vote -> 93.8% Seats", 
               fontsize=9, fontweight='bold', color='#7570b3')

    ax_sg.scatter([sg_metrics[2020]['Mean_PAP']], [89.2], color='#d95f02', s=100, edgecolor='black', zorder=6)
    ax_sg.text(sg_metrics[2020]['Mean_PAP'] - 12, 88.0, f"2020 Actual:\n{sg_metrics[2020]['Mean_PAP']:.1f}% Vote -> 89.2% Seats", 
               fontsize=9, fontweight='bold', color='#d95f02')

    ax_sg.set_title('Singapore: Extreme Winner Bonus & The 2/3 Supermajority Cliff', fontsize=12, fontweight='bold')
    ax_sg.set_xlabel('National PAP Popular Vote Share (%)', fontsize=11, fontweight='bold')
    ax_sg.set_ylabel('Parliamentary Seat Share Won (%)', fontsize=11, fontweight='bold')
    ax_sg.set_xlim(35, 80)
    ax_sg.set_ylim(0, 105)
    ax_sg.legend(loc='lower right', frameon=True, fontsize=8.5)

    # Panel 2: US House
    us_votes = np.linspace(40, 65, 200)
    us_cont = df_us_merged[df_us_merged['Status_2026'] == 'Contested'].copy()
    cur_dem_mean = us_metrics['Mean_Dem']
    cur_dem_margins = us_cont['Margin_2026'].values
    unc_dem_seats = (df_us_merged['Winner_2026'] == 'Democrat').sum() - (us_cont['Winner_2026'] == 'Democrat').sum()

    us_seat_pcts = []
    for target_v in us_votes:
        swing_margin = (target_v - cur_dem_mean) * 2.0
        won = unc_dem_seats + sum(1 for m in cur_dem_margins if (m + swing_margin) > 0.0)
        us_seat_pcts.append((won / 435.0) * 100.0)

    ax_us.plot(us_votes, us_votes, color='#999999', linestyle=':', linewidth=1.5, label='Perfect Proportionality (1:1)')
    ax_us.axvspan(47.5, 52.5, color='#fee8e8', alpha=0.5, label='Cracking Zone (132 House Seats clustered +/-5%)')
    ax_us.plot(us_votes, us_seat_pcts, color='#2b5c8f', linewidth=2.8, label='US House 2026 Seats-Votes Curve')

    ax_us.axhline(50.0, color='#08519c', linestyle='-.', linewidth=2.0)
    ax_us.text(40.5, 51.0, '50% House Majority Threshold (218 Seats)', color='#08519c', fontweight='bold', fontsize=9.5)

    ax_us.axhline(66.67, color='#b30000', linestyle='--', linewidth=2.0)
    ax_us.text(40.5, 67.5, '2/3 House Supermajority Threshold (290 Seats)', color='#b30000', fontweight='bold', fontsize=9.5)

    v_flip_us = us_metrics['Vote_Share_Flip_Majority']
    ax_us.axvline(v_flip_us, color='#08519c', linestyle=':', linewidth=1.8)

    ax_us.annotate(f'Majority Flips at {v_flip_us:.1f}% Dem Vote\n(Swing needed: +{us_metrics["Swing_Flip_Majority"]:.2f}% to GOP)',
                   xy=(v_flip_us, 50.0), xytext=(v_flip_us - 7.0, 38),
                   arrowprops=dict(facecolor='#08519c', shrink=0.08, width=1.5, headwidth=6),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#eff6ff', edgecolor='#08519c', alpha=0.95),
                   fontsize=9, fontweight='bold', color='#08519c')

    act_dem_count = us_metrics['Dem_Seats_2026']
    act_dem_pct = act_dem_count / 435.0 * 100.0
    ax_us.scatter([cur_dem_mean], [act_dem_pct], color='#2b5c8f', s=120, edgecolor='black', zorder=6)
    ax_us.text(cur_dem_mean - 1.5, act_dem_pct + 3.0, f"2026 Actual:\n{cur_dem_mean:.1f}% Vote -> {act_dem_pct:.1f}% Seats ({act_dem_count})", 
               fontsize=9, fontweight='bold', color='#2b5c8f')

    ax_us.set_title('US House: Bipolar Polarization & Competitive Majority Tipping Point', fontsize=12, fontweight='bold')
    ax_us.set_xlabel('Democratic National Popular Vote Share (%)', fontsize=11, fontweight='bold')
    ax_us.set_ylabel('House Seat Share Won (%)', fontsize=11, fontweight='bold')
    ax_us.set_xlim(40, 65)
    ax_us.set_ylim(0, 105)
    ax_us.legend(loc='lower right', frameon=True, fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_path}")

def plot_singapore_cascade(sg_metrics, output_path='singapore_cracking_cascade.png'):
    fig, (ax20, ax25) = plt.subplots(1, 2, figsize=(18, 10), sharex=True)
    fig.suptitle('The Singapore Seat Cascade: Flip Sequences & Threshold Fragility (2020 vs 2025)\n(Prominently Highlighting Aljunied, East Coast, West Coast & Tanjong Pagar; Zero-Collision Labels)', 
                 fontsize=14, fontweight='bold', y=0.98)

    def plot_cascade_panel(ax, yr, title):
        df_w = sg_metrics[yr]['Ward_DF'].copy()
        x_vals = -(df_w['PAP_Share'] - 50.0)
        
        colors_bar = []
        edge_colors = []
        lw_list = []
        
        for idx, r in df_w.iterrows():
            c_name = r['Constituency']
            t_key = match_target_ward(c_name)
            
            # Base color by threshold
            if not r['Won_By_PAP']:
                base_c = '#2b5c8f'
            elif r['Swing_Needed_To_Flip'] <= 5.0:
                base_c = '#d73027'
            elif r['Constituency'] == sg_metrics[yr]['Tipping_Ward_23']:
                base_c = '#7f0000'
            elif r['Swing_Needed_To_Flip'] <= sg_metrics[yr]['Swing_Break_23']:
                base_c = '#fc8d59'
            elif r['Constituency'] == sg_metrics[yr]['Tipping_Ward_50']:
                base_c = '#542788'
            elif r['Swing_Needed_To_Flip'] <= sg_metrics[yr]['Swing_Break_50']:
                base_c = '#fee090'
            else:
                base_c = '#bdbdbd'
                
            colors_bar.append(base_c)
            
            # Highlight target wards with Gold border
            if t_key:
                edge_colors.append('#ffd700')
                lw_list.append(2.4)
            else:
                edge_colors.append('black')
                lw_list.append(0.5)

        bars = ax.barh(df_w['Constituency'], x_vals, color=colors_bar, edgecolor=edge_colors, linewidth=lw_list, height=0.7)
        ax.axvline(0, color='black', linewidth=1.5, linestyle='-')
        
        swing_23 = sg_metrics[yr]['Swing_Break_23']
        swing_50 = sg_metrics[yr]['Swing_Break_50']
        ax.axvline(-swing_23, color='#7f0000', linestyle='--', linewidth=1.8, label=f'2/3 Breaker (-{swing_23:.1f}% PAP Margin)')
        ax.axvline(-swing_50, color='#542788', linestyle='-.', linewidth=1.8, label=f'50% Breaker (-{swing_50:.1f}% PAP Margin)')
        ax.axvspan(-5.0, 0, color='#fee8e8', alpha=0.45, label='Cracking Zone (<= 5% Swing)')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(-35, 15)
        
        ax.set_yticks(range(len(df_w)))
        yticklabels = []
        for c in df_w['Constituency']:
            t_key = match_target_ward(c)
            if t_key:
                yticklabels.append(f"★ {c}")
            else:
                yticklabels.append(c)
        ax.set_yticklabels(yticklabels, fontsize=8.2)
        
        for bar, (_, r), val in zip(bars, df_w.iterrows(), x_vals):
            w_width = bar.get_width()
            t_key = match_target_ward(r['Constituency'])
            offset = -0.5 if w_width < 0 else 0.5
            ha = 'right' if w_width < 0 else 'left'
            
            if r['Won_By_PAP']:
                txt = f"{r['MPs']} MPs ({r['PAP_Share']:.1f}%)"
            else:
                txt = f"{r['MPs']} MPs ({100-r['PAP_Share']:.1f}% Opp)"
                
            txt_color = '#b30000' if t_key else '#222222'
            font_w = 'bold' if t_key else 'normal'
            
            ax.text(w_width + offset, bar.get_y() + bar.get_height()/2.0, txt, 
                    va='center', ha=ha, fontsize=7.8, fontweight=font_w, color=txt_color,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.15))
        
        ax.legend(loc='lower left', frameon=True, fontsize=8)

    plot_cascade_panel(ax20, 2020, f"Singapore GE 2020 (93 MPs)\n2/3 Tipping: {sg_metrics[2020]['Tipping_Ward_23']} (+{sg_metrics[2020]['Swing_Break_23']:.1f}%) | 50% Tipping: {sg_metrics[2020]['Tipping_Ward_50']} (+{sg_metrics[2020]['Swing_Break_50']:.1f}%)")
    plot_cascade_panel(ax25, 2025, f"Singapore GE 2025 (97 MPs)\n2/3 Tipping: {sg_metrics[2025]['Tipping_Ward_23']} (+{sg_metrics[2025]['Swing_Break_23']:.1f}%) | 50% Tipping: {sg_metrics[2025]['Tipping_Ward_50']} (+{sg_metrics[2025]['Swing_Break_50']:.1f}%)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_path} (Highlighting Target Wards)")

def plot_case_study_2006_2011(output_path='singapore_case_study_2006_2011.png'):
    """
    Plots the 2006-2011 Case Study for Aljunied, East Coast, West Coast, and Tanjong Pagar vs National Swing.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Case Study (2006–2011): Aljunied, East Coast, West Coast & Tanjong Pagar vs. National Swing\n(Deconstructing the Asymmetric Impact of Singapore\'s 2011 Watershed Election)', 
                 fontsize=13, fontweight='bold', y=0.98)

    case_data = [
        {'Constituency': 'Aljunied GRC', 'PAP_2006': 56.09, 'PAP_2011': 45.28, 'Swing_PAP': -10.81, 'Rel_Swing_vs_Nat': -4.35, 'Status': 'Flipped to WP'},
        {'Constituency': 'East Coast GRC', 'PAP_2006': 63.86, 'PAP_2011': 54.83, 'Swing_PAP': -9.03, 'Rel_Swing_vs_Nat': -2.57, 'Status': 'Cracking Zone'},
        {'Constituency': 'National Benchmark', 'PAP_2006': 66.60, 'PAP_2011': 60.14, 'Swing_PAP': -6.46, 'Rel_Swing_vs_Nat': 0.00, 'Status': 'National Baseline'},
        {'Constituency': 'West Coast GRC', 'PAP_2006': np.nan, 'PAP_2011': 66.57, 'Swing_PAP': np.nan, 'Rel_Swing_vs_Nat': +6.43, 'Status': 'Western Fortress'},
        {'Constituency': 'Tanjong Pagar GRC', 'PAP_2006': np.nan, 'PAP_2011': np.nan, 'Swing_PAP': 0.00, 'Rel_Swing_vs_Nat': np.nan, 'Status': 'Walkover Firewall'}
    ]

    df_case = pd.DataFrame(case_data)
    x = np.arange(len(df_case))
    width = 0.35

    v06_vals = [r['PAP_2006'] if pd.notnull(r['PAP_2006']) else 0 for _, r in df_case.iterrows()]
    bars1 = ax1.bar(x - width/2, v06_vals, width, label='2006 PAP Vote Share (%)', color='#9ecae1', edgecolor='black', linewidth=0.6)

    v11_vals = [r['PAP_2011'] if pd.notnull(r['PAP_2011']) else 0 for _, r in df_case.iterrows()]
    bars2 = ax1.bar(x + width/2, v11_vals, width, label='2011 PAP Vote Share (%)', color='#fc9272', edgecolor='black', linewidth=0.6)

    ax1.axhline(50.0, color='black', linestyle=':', linewidth=1.5, label='50% Win Parity')
    ax1.set_ylabel('PAP Vote Share (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Direct Vote Share Comparison: 2006 vs. 2011', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_case['Constituency'], fontsize=9.5, fontweight='bold', rotation=15)
    ax1.set_ylim(0, 85)
    ax1.legend(loc='lower left', frameon=True, fontsize=8.5)

    for idx, r in df_case.iterrows():
        if pd.notnull(r['PAP_2006']):
            ax1.text(idx - width/2, r['PAP_2006'] + 1.2, f"{r['PAP_2006']:.1f}%", ha='center', fontsize=8.5, fontweight='bold', color='#1f77b4')
        else:
            ax1.text(idx - width/2, 2.5, "Walkover\n(100% Seats)", ha='center', fontsize=7.5, color='#555555', style='italic')
            
        if pd.notnull(r['PAP_2011']):
            ax1.text(idx + width/2, r['PAP_2011'] + 1.2, f"{r['PAP_2011']:.1f}%", ha='center', fontsize=8.5, fontweight='bold', color='#d62728')
        else:
            ax1.text(idx + width/2, 2.5, "Walkover\n(100% Seats)", ha='center', fontsize=7.5, color='#555555', style='italic')

    rel_swings = []
    rel_colors = []
    for _, r in df_case.iterrows():
        val = r['Rel_Swing_vs_Nat']
        if pd.notnull(val):
            rel_swings.append(val)
            rel_colors.append('#2b5c8f' if val < 0 else ('#b30000' if val > 0 else '#888888'))
        else:
            rel_swings.append(0.0)
            rel_colors.append('#cccccc')

    bars_rel = ax2.barh(df_case['Constituency'], rel_swings, color=rel_colors, edgecolor='black', linewidth=0.7, height=0.6)
    ax2.axvline(0, color='black', linewidth=1.5, linestyle='-')
    ax2.axvspan(-8, 0, color='#e8f0fe', alpha=0.35, label='Swung Harder to Opposition than National Avg')
    ax2.axvspan(0, 8, color='#fee8e8', alpha=0.35, label='PAP Outperformed National Baseline')

    ax2.set_xlabel('Relative Performance vs. National Swing (% Points)\n[<-- Swung Harder to Opp (Right) | PAP Outperformed National Baseline (Left) -->]', 
                  fontsize=10.5, fontweight='bold')
    ax2.set_title('Swing Relative to the National Tide (-6.46% PAP Drop)', fontsize=12, fontweight='bold')
    ax2.set_xlim(-8, 8)

    for bar, (_, r) in zip(bars_rel, df_case.iterrows()):
        val = r['Rel_Swing_vs_Nat']
        y_pos = bar.get_y() + bar.get_height()/2.0
        if pd.notnull(val) and val != 0.0:
            offset = 0.25 if val > 0 else -0.25
            ha = 'left' if val > 0 else 'right'
            sign = '+' if val > 0 else ''
            ax2.text(val + offset, y_pos, f"{sign}{val:.2f}% ({r['Status'].split('(')[0].strip()})", 
                     va='center', ha=ha, fontsize=8.2, fontweight='bold', color='#111111')
        elif r['Constituency'] == 'National Benchmark':
            ax2.text(0.25, y_pos, 'Baseline (-6.46% Swing)', va='center', ha='left', fontsize=8.2, fontweight='bold', color='#b30000')
        else:
            ax2.text(0.25, y_pos, 'Uncontested (Insulated from Swing)', va='center', ha='left', fontsize=8.0, style='italic', color='#555555')

    ax2.legend(loc='lower right', frameon=True, fontsize=8.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_path}")

if __name__ == '__main__':
    df_const, df_sum = load_and_process_singapore_data()
    df_us, df_us_sum = load_and_merge_us_data()
    print("--- Singapore Summary (PAP on Left) ---")
    print(df_sum[['Year', 'Contested_Wards', 'National_PAP_Pct', 'National_PAP_Lean_Pct', 'National_Coord_Pct']])
    print("--- US Data Integrated Summary (GOP on Left, Dem on Right) ---")
    print(df_us_sum)
    plot_faceted_distribution_same_scale(df_const, df_sum)
    plot_trend_and_overlay(df_const, df_sum)
    plot_three_panel_comparison(df_const, df_sum, df_us, df_us_sum)
    plot_case_study_2006_2011()
    
    # Dummymandering metrics & graphs
    sg_met, us_met = compute_dummymandering_metrics(df_us)
    plot_seats_votes_curves(sg_met, us_met, df_us)
    plot_singapore_cascade(sg_met)

