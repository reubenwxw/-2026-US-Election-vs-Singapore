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
        2011: 'Singapore_2011_results.csv',
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

def plot_faceted_distribution_same_scale(df_constituencies, df_summary, output_path='singapore_partisan_lean_same_scale.png'):
    SCALE_MIN = -35.0  # PAP on Left
    SCALE_MAX = 20.0   # Opp on Right
    years = [2011, 2020, 2025]
    colors = {'2011': '#2b5c8f', '2020': '#d95f02', '2025': '#7570b3'}
    
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle('Distribution of Constituency Partisan Lean Across Singapore General Elections\n(PAP on the LEFT, Opposition on the RIGHT on a Standardized Shared Scale)', 
                 fontsize=15, fontweight='bold', y=0.98)
    
    for i, year in enumerate(years):
        ax = axes[i]
        year_data = df_constituencies[(df_constituencies['Year'] == year) & (df_constituencies['Contested'])]
        sum_row = df_summary[df_summary['Year'] == year].iloc[0]
        
        nat_coord = sum_row['National_Coord_Pct']
        mean_coord = sum_row['Mean_Ward_Coord_Pct']
        color = colors[str(year)]
        
        ax.axvspan(SCALE_MIN, 0, color='#fee8e8', alpha=0.45, label='PAP Lean (Ruling Party)' if i == 0 else "")
        ax.axvspan(0, SCALE_MAX, color='#e8f0fe', alpha=0.45, label='Opposition Lean' if i == 0 else "")
        ax.axvline(0, color='#444444', linestyle=':', linewidth=1.4, alpha=0.8, label='50-50 Parity' if i == 0 else "")
        
        sns.kdeplot(year_data['Partisan_Coord_Pct'], ax=ax, color=color, fill=True, alpha=0.25, linewidth=2.2, bw_adjust=0.8)
        
        np.random.seed(42)
        jitter = np.random.normal(0, 0.003, size=len(year_data))
        ax.scatter(year_data['Partisan_Coord_Pct'], jitter + 0.012, 
                   color=color, edgecolor='white', s=55, alpha=0.85, zorder=4,
                   label=f'{year} Constituencies (n={len(year_data)})')
        
        min_ward = year_data.loc[year_data['Partisan_Coord_Pct'].idxmin()]
        max_ward = year_data.loc[year_data['Partisan_Coord_Pct'].idxmax()]
        ax.text(min_ward['Partisan_Coord_Pct'], 0.022, f" {min_ward['Constituency']} (PAP {min_ward['PAP_Share_Pct']:.1f}%)", 
                fontsize=8.5, ha='right', color='#8b0000', fontweight='bold')
        ax.text(max_ward['Partisan_Coord_Pct'], 0.022, f" {max_ward['Constituency']} (Opp {max_ward['Opp_Share_Pct']:.1f}%)", 
                fontsize=8.5, ha='left', color='#1a3a6b', fontweight='bold')
        
        ax.axvline(nat_coord, color='#b30000', linestyle='--', linewidth=2.5, zorder=5,
                   label=f'Avg PAP Lean: +{abs(nat_coord):.2f}%')
        ax.axvline(mean_coord, color='#e66101', linestyle='-.', linewidth=1.8, alpha=0.85, zorder=5,
                   label=f'Mean Ward Coord: {mean_coord:.2f}%')
        
        ax.annotate(
            f"Avg PAP Lean: +{abs(nat_coord):.2f}%\n(PAP {sum_row['National_PAP_Pct']:.2f}%)",
            xy=(nat_coord, 0.038),
            xytext=(nat_coord - 7.5, 0.038),
            arrowprops=dict(facecolor='#b30000', shrink=0.08, width=1.5, headwidth=6, alpha=0.9),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#b30000', alpha=0.95),
            fontsize=9.5, fontweight='bold', color='#800000', zorder=6
        )
        
        ax.set_ylabel(f'{year} Density', fontsize=11, fontweight='bold')
        ax.set_xlim(SCALE_MIN, SCALE_MAX)
        ax.set_ylim(-0.005, 0.065)
        ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8.5)

    axes[-1].set_xlabel('Partisan Lean Axis (% relative to 50% Competitive Parity)\n[<-- PAP Leaning (Ruling Party) | 50-50 Parity | Opposition Leaning -->]', 
                       fontsize=12, fontweight='bold', labelpad=10)

    ticks = np.arange(SCALE_MIN, SCALE_MAX + 1, 10)
    def format_tick(t):
        if t < 0: return f"PAP +{abs(t):.0f}%\n({50-t:.0f}% PAP)"
        elif t > 0: return f"Opp +{t:.0f}%\n({50-t:.0f}% PAP)"
        return "0%\n(Parity)"
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([format_tick(t) for t in ticks], fontsize=9.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_path} (PAP on Left)")

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
    for yr, fpath in [(2020, 'Singapore_2020_results.csv'), (2025, 'Singapore_2025_results.csv')]:
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

    seats_2020 = simulate_sg_curve(2020)
    seats_2025 = simulate_sg_curve(2025)

    ax_sg.plot(sim_votes, sim_votes, color='#999999', linestyle=':', linewidth=1.5, label='Perfect Proportionality (1:1)')
    ax_sg.axvspan(48, 56, color='#fee8e8', alpha=0.5, label='Cracking Cascade Zone (High Disproportionality)')
    ax_sg.plot(sim_votes, seats_2020, color='#d95f02', linewidth=2.6, label='Singapore 2020 Seats-Votes Curve')
    ax_sg.plot(sim_votes, seats_2025, color='#7570b3', linewidth=2.8, label='Singapore 2025 Seats-Votes Curve')

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
    """
    Plots the Singapore 2025 Seat Cascade Waterfall chart.
    Standardized Bipolar Orientation: PAP on LEFT (Negative / Red), Opposition on RIGHT (Positive / Blue).
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('The Singapore 2025 Seat Cascade: Key Swings Required to Flip Wards & Break Thresholds\n(PAP on the LEFT, Opposition on the RIGHT | GRC Block Voting Waterfall)', 
                 fontsize=13, fontweight='bold', y=0.98)

    df_w25 = sg_metrics[2025]['Ward_DF'].copy()
    x_values = -(df_w25['PAP_Share'] - 50.0)

    colors_bar = []
    for idx, r in df_w25.iterrows():
        if not r['Won_By_PAP']:
            colors_bar.append('#2b5c8f')  # Won by Opposition (Hougang, Aljunied, Sengkang) - Blue
        elif r['Swing_Needed_To_Flip'] <= 5.0:
            colors_bar.append('#d73027')  # Volatile Cracking Zone (<= 5% swing) - Bright Red
        elif r['Constituency'] == sg_metrics[2025]['Tipping_Ward_23']:
            colors_bar.append('#7f0000')  # 2/3 Supermajority Breaker - Deep Maroon
        elif r['Swing_Needed_To_Flip'] <= sg_metrics[2025]['Swing_Break_23']:
            colors_bar.append('#fc8d59')  # Flips before 2/3 - Orange
        elif r['Constituency'] == sg_metrics[2025]['Tipping_Ward_50']:
            colors_bar.append('#542788')  # 50% Majority Breaker - Purple
        elif r['Swing_Needed_To_Flip'] <= sg_metrics[2025]['Swing_Break_50']:
            colors_bar.append('#fee090')  # Flips before 50% - Gold
        else:
            colors_bar.append('#bdbdbd')  # Safe PAP Core - Gray

    bars = ax.barh(df_w25['Constituency'], x_values, color=colors_bar, edgecolor='black', linewidth=0.6, height=0.7)

    ax.axvline(0, color='black', linewidth=1.5, linestyle='-')

    swing_23 = sg_metrics[2025]['Swing_Break_23']
    swing_50 = sg_metrics[2025]['Swing_Break_50']
    ax.axvline(-swing_23, color='#7f0000', linestyle='--', linewidth=2.0, label=f'2/3 Supermajority Threshold (-{swing_23:.2f}% PAP Margin)')
    ax.axvline(-swing_50, color='#542788', linestyle='-.', linewidth=2.0, label=f'50% Simple Majority Threshold (-{swing_50:.2f}% PAP Margin)')

    ax.axvspan(-5.0, 0, color='#fee8e8', alpha=0.45, label='Volatile Cracking Zone (<= 5% Swing: Jalan Kayu, Tampines, Sembawang West)')

    ax.set_xlabel('[<-- PAP Strongholds (Swing Required to Flip to Opposition) | Won by Opposition -->]\nMargin from 50% Win Parity (Percentage Points)', 
                  fontsize=11, fontweight='bold')
    ax.set_xlim(-35, 15)

    for bar, (_, r), val in zip(bars, df_w25.iterrows(), x_values):
        w_width = bar.get_width()
        offset = -0.5 if w_width < 0 else 0.5
        ha = 'right' if w_width < 0 else 'left'
        txt = f"{r['MPs']} MPs ({r['PAP_Share']:.1f}% PAP)" if r['Won_By_PAP'] else f"{r['MPs']} MPs ({100-r['PAP_Share']:.1f}% Opp)"
        ax.text(w_width + offset, bar.get_y() + bar.get_height()/2.0, txt, 
                va='center', ha=ha, fontsize=8, fontweight='bold', color='#222222')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2b5c8f', label='Already Opposition Held (10 MPs: Hougang, Aljunied, Sengkang)'),
        Patch(facecolor='#d73027', label='Volatile Cracking Zone (<= 5% Swing: 7 MPs)'),
        Patch(facecolor='#fc8d59', label='Flips to Break 2/3 Supermajority (<= 11.38% Swing: 23 MPs)'),
        Patch(facecolor='#7f0000', label='Tipping Point Seat for 2/3 (Bukit Panjang)'),
        Patch(facecolor='#fee090', label='Flips to Break 50% Majority (<= 18.49% Swing: 16 MPs)'),
        Patch(facecolor='#542788', label='Tipping Point Seat for 50% (Kebun Baru)'),
        Patch(facecolor='#bdbdbd', label='Safe PAP Heartland (> 18.5% Swing: 48 MPs)')
    ]
    ax.legend(handles=legend_elements, loc='lower left', frameon=True, fontsize=8.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {output_path} (PAP on the LEFT)")

if __name__ == '__main__':
    df_const, df_sum = load_and_process_singapore_data()
    df_us, df_us_sum = load_and_merge_us_data()
    print("--- Singapore Summary (PAP on Left) ---")
    print(df_sum[['Year', 'Contested_Wards', 'National_PAP_Pct', 'National_PAP_Lean_Pct', 'National_Coord_Pct']])
    print("--- US Data Integrated Summary (GOP on Left, Dem on Right) ---")
    print(df_us_sum)
    plot_faceted_distribution_same_scale(df_const, df_sum)
    plot_three_panel_comparison(df_const, df_sum, df_us, df_us_sum)
    
    # Dummymandering metrics & graphs
    sg_met, us_met = compute_dummymandering_metrics(df_us)
    plot_seats_votes_curves(sg_met, us_met, df_us)
    plot_singapore_cascade(sg_met)

