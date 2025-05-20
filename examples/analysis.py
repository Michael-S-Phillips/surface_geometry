# analysis.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable
import scipy.stats as stats
from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import sys
from PIL import Image

# Add parent directory to path
sys.path.append('..')

# Import functions from our package
from surface_geometry.functions import D_func, R_func, HL0_func

# Create output directories
os.makedirs("figs", exist_ok=True)

# Scope and grain parameters
L = 2      # Extent, 2 by 2 m reef patches
L0 = 2/32  # Grain, resolution of processing ~ 6 cm

# Load data
print("Loading data...")
dat = pd.read_csv("output/master_20200709.csv")

# Create polynomial fit function for use in visualizations
def poly_fit(x, y, degree=2):
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    x_new = np.linspace(min(x), max(x), 100)
    y_new = poly(x_new)
    return x_new, y_new, coeffs

# Fig 2: Surface descriptor associations
print("Creating Figure 2: Surface descriptor associations...")
fig = plt.figure(figsize=(10, 10))

# Create grid for layout
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

# Plot 1: R vs D
ax1 = plt.subplot(gs[0, 0])
ax1.scatter(dat['D_theory'], dat['R2_log10'], color='black', alpha=0.3)
ax1.set_xlabel(r'Fractal dimension ($D$)')
ax1.set_ylabel(r'Rugosity ($R^2 - 1$)')
ax1.set_ylim(-2, 0.7)
ax1.set_xlim(2, 2.6)
ax1.set_yticks(np.log10([0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]))
ax1.set_yticklabels(['0.01', '0.05', '0.1', '0.2', '0.5', '1', '2', '5'])
ax1.set_xticks([2, 2.2, 2.4, 2.6])
r_squared = np.corrcoef(dat['D_theory'], dat['R2_log10'])[0, 1]**2
ax1.text(2.1, 0.6, f"$r^2 = {r_squared:.3f}$")
ax1.text(0.05, 0.95, 'a', transform=ax1.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top')

# Create convex hull for R vs D
points = np.column_stack([dat['D_theory'], dat['R2_log10']])
hull = ConvexHull(points)
for simplex in hull.simplices:
    ax1.plot(points[simplex, 0], points[simplex, 1], 'k--', lw=0.5)

# Plot 2: R vs HL0
ax2 = plt.subplot(gs[0, 1])
ax2.scatter(dat['HL0_log10'], dat['R2_log10'], color='black', alpha=0.3)
ax2.set_xlabel(r'Height range ($\Delta H / \sqrt{2} L_0$)')
ax2.set_ylabel(r'Rugosity ($R^2 - 1$)')
ax2.set_xlim(np.log10(2), np.log10(50))
ax2.set_ylim(-2, 0.7)
ax2.set_xticks(np.log10([2, 5, 10, 20, 50]))
ax2.set_xticklabels(['2', '5', '10', '20', '50'])
ax2.set_yticks(np.log10([0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]))
ax2.set_yticklabels(['0.01', '0.05', '0.1', '0.2', '0.5', '1', '2', '5'])
r_squared = np.corrcoef(dat['HL0_log10'], dat['R2_log10'])[0, 1]**2
ax2.text(0.5, 0.6, f"$r^2 = {r_squared:.3f}$")
ax2.text(0.05, 0.95, 'b', transform=ax2.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top')

# Create convex hull for R vs HL0
points = np.column_stack([dat['HL0_log10'], dat['R2_log10']])
hull = ConvexHull(points)
for simplex in hull.simplices:
    ax2.plot(points[simplex, 0], points[simplex, 1], 'k--', lw=0.5)

# Plot 3: HL0 vs D
ax3 = plt.subplot(gs[1, 0])
ax3.scatter(dat['D_theory'], dat['HL0_log10'], color='black', alpha=0.3)
ax3.set_xlabel(r'Fractal dimension ($D$)')
ax3.set_ylabel(r'Height range ($\Delta H / \sqrt{2} L_0$)')
ax3.set_ylim(np.log10(2), np.log10(50))
ax3.set_xlim(2, 2.6)
ax3.set_yticks(np.log10([2, 5, 10, 20, 50]))
ax3.set_yticklabels(['2', '5', '10', '20', '50'])
ax3.set_xticks([2, 2.2, 2.4, 2.6])
r_squared = np.corrcoef(dat['HL0_log10'], dat['D_theory'])[0, 1]**2
ax3.text(2.4, np.log10(40), f"$r^2 = {r_squared:.3f}$")
ax3.text(0.05, 0.95, 'c', transform=ax3.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top')

# Create convex hull for HL0 vs D
points = np.column_stack([dat['D_theory'], dat['HL0_log10']])
hull = ConvexHull(points)
for simplex in hull.simplices:
    ax3.plot(points[simplex, 0], points[simplex, 1], 'k--', lw=0.5)

# Plot 4: 3D Surface plot showing the relationship between R, D, and H
ax4 = plt.subplot(gs[1, 1], projection='3d')

# Create a grid for D and R
grid_lines = 100
R_pred = np.linspace(min(dat['R2_log10'])-0.25, max(dat['R2_log10'])+0.25, grid_lines)
H_pred = np.linspace(min(dat['HL0_log10'])-0.25, max(dat['HL0_log10'])+0.25, grid_lines)
R_mesh, H_mesh = np.meshgrid(R_pred, H_pred)

# Calculate D for each R,H combination
D_mat = np.zeros(R_mesh.shape)
for i in range(grid_lines):
    for j in range(grid_lines):
        D_mat[i, j] = D_func((np.sqrt(2) * L0)*(10**H_mesh[i, j]), 
                            np.sqrt(10**R_mesh[i, j]+1), L, L0)

# Filter values outside appropriate range
D_mat[D_mat < 2] = np.nan
D_mat[D_mat > 2.6] = np.nan

# Plot the surface
surf = ax4.plot_surface(R_mesh, H_mesh, D_mat, cmap='Reds', alpha=0.3, antialiased=True)

# Add data points
scatter = ax4.scatter(dat['R2_log10'], dat['HL0_log10'], dat['D_theory'], 
                     c=dat['D_theory'], cmap='Reds_r', s=20, alpha=0.8)

# Add a color bar
cbar = plt.colorbar(scatter, ax=ax4, shrink=0.5, label='$D$')

# Set labels and view
ax4.set_xlabel('Rugosity')
ax4.set_ylabel('Height range')
ax4.set_zlabel('Fractal dimension')
ax4.view_init(elev=20, azim=45)

# Calculate and display r^2 for theory vs empirical D
ss_tot = np.sum((dat['D'] - np.mean(dat['D_theory']))**2)
ss_res = np.sum((dat['D'] - dat['D_theory'])**2)
r_squared_3d = 1 - (ss_res / ss_tot)
ax4.text2D(0.05, 0.95, f"$r^2 = {r_squared_3d:.3f}$", transform=ax4.transAxes, 
          fontsize=10)
ax4.text2D(0.05, 0.9, 'd', transform=ax4.transAxes, fontsize=12, fontweight='bold',
          verticalalignment='top')

plt.tight_layout()
plt.savefig("figs/fig2.pdf", dpi=300)
plt.close()

# Extract megaplot data for biodiversity analysis
print("Analyzing biodiversity relationships...")
megaplot = dat[dat['site'] == 'megaplot'].copy()
megaplot['abd_sqrt'] = np.sqrt(megaplot['abd'])
megaplot['spp_sqrt'] = np.sqrt(megaplot['spp'])
megaplot['pie_asin'] = np.arcsin(megaplot['pie'])

# Subset for analyses
megaplotA = megaplot[megaplot['abd'] > 0].copy()  # Test results removing zero abundance patches
megaplotR = megaplot[megaplot['R2_log10'] < 0.2].copy()  # Test results removing high rugosity patches

# Create Figure S7 with geometry-biodiversity relationships
fig, axs = plt.subplots(3, 3, figsize=(12, 12))

# Helper function for GAM-like smoothing using Gaussian Process Regression
def gp_smooth(x, y, x_pred, return_model=False):
    # Define kernel with both RBF and noise components
    kernel = 1.0 * RBF(length_scale=0.2) + WhiteKernel(noise_level=0.1)
    
    # Create and fit GPR model
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10)
    gpr.fit(x.reshape(-1, 1), y)
    
    # Make predictions
    y_pred, sigma = gpr.predict(x_pred.reshape(-1, 1), return_std=True)
    
    if return_model:
        return y_pred, sigma, gpr
    return y_pred, sigma

# Function to calculate R^2 for GAM fit
def gam_r2(x, y, model):
    y_pred = model.predict(x.reshape(-1, 1))
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum((y - y_pred)**2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

# Generate prediction points
x_R = np.linspace(megaplot['R2_log10'].min(), megaplot['R2_log10'].max(), 100)
x_R2 = np.linspace(megaplotR['R2_log10'].min(), megaplotR['R2_log10'].max(), 100)
x_D = np.linspace(megaplot['D_theory'].min(), megaplot['D_theory'].max(), 100)
x_H = np.linspace(megaplot['HL0_log10'].min(), megaplot['HL0_log10'].max(), 100)

# PIE plots
# PIE vs R
ax = axs[0, 0]
ax.scatter(megaplot['R2_log10'], megaplot['pie_asin'], color='grey')
y_pred, sigma, model = gp_smooth(megaplot['R2_log10'].values, megaplot['pie_asin'].values, x_R, return_model=True)
ax.plot(x_R, y_pred, 'k--')
ax.fill_between(x_R, y_pred - 2*sigma, y_pred + 2*sigma, color='black', alpha=0.2)

# Also add polynomial fit
x_poly, y_poly, _ = poly_fit(megaplot['R2_log10'], megaplot['pie_asin'])
ax.plot(x_poly, y_poly, 'b-')
ax.fill_between(x_poly, y_poly - 0.1, y_poly + 0.1, color='blue', alpha=0.2)

# Add subset model
y_pred_subset = gp_smooth(megaplotR['R2_log10'].values, megaplotR['pie_asin'].values, x_R2)[0]
ax.plot(x_R2, y_pred_subset, 'r-')
ax.fill_between(x_R2, y_pred_subset - 0.1, y_pred_subset + 0.1, color='red', alpha=0.2)

r2 = gam_r2(megaplot['R2_log10'].values, megaplot['pie_asin'].values, model)
ax.set_xlabel(r'$R^2 - 1$ (log scale)')
ax.set_ylabel('Probability of interspecific encounter')
ax.text(0.05, 0.95, 'a', transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top')
ax.text(0.7, 0.95, f"$r^2 = {r2:.3f}$", transform=ax.transAxes, fontsize=10)
ax.legend(['GAM', 'LM (poly)', 'GAM (subset R)'], loc='lower right')

# PIE vs D
ax = axs[0, 1]
ax.scatter(megaplot['D_theory'], megaplot['pie_asin'], color='grey')
y_pred, sigma, model = gp_smooth(megaplot['D_theory'].values, megaplot['pie_asin'].values, x_D, return_model=True)
ax.plot(x_D, y_pred, 'k--')
ax.fill_between(x_D, y_pred - 2*sigma, y_pred + 2*sigma, color='black', alpha=0.2)

# Add polynomial fit
x_poly, y_poly, _ = poly_fit(megaplot['D_theory'], megaplot['pie_asin'])
ax.plot(x_poly, y_poly, 'b-')
ax.fill_between(x_poly, y_poly - 0.1, y_poly + 0.1, color='blue', alpha=0.2)

r2 = gam_r2(megaplot['D_theory'].values, megaplot['pie_asin'].values, model)
ax.set_xlabel('D')
ax.set_ylabel('Probability of interspecific encounter')
ax.text(0.05, 0.95, 'b', transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top')
ax.text(0.7, 0.95, f"$r^2 = {r2:.3f}$", transform=ax.transAxes, fontsize=10)

# PIE vs H
ax = axs[0, 2]
ax.scatter(megaplot['HL0_log10'], megaplot['pie_asin'], color='grey')
y_pred, sigma, model = gp_smooth(megaplot['HL0_log10'].values, megaplot['pie_asin'].values, x_H, return_model=True)
ax.plot(x_H, y_pred, 'k--')
ax.fill_between(x_H, y_pred - 2*sigma, y_pred + 2*sigma, color='black', alpha=0.2)

# Add polynomial fit
x_poly, y_poly, _ = poly_fit(megaplot['HL0_log10'], megaplot['pie_asin'])
ax.plot(x_poly, y_poly, 'b-')
ax.fill_between(x_poly, y_poly - 0.1, y_poly + 0.1, color='blue', alpha=0.2)

r2 = gam_r2(megaplot['HL0_log10'].values, megaplot['pie_asin'].values, model)
ax.set_xlabel(r'$\Delta H / \sqrt{2} L_0$ (log scale)')
ax.set_ylabel('Probability of interspecific encounter')
ax.text(0.05, 0.95, 'c', transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top')
ax.text(0.7, 0.95, f"$r^2 = {r2:.3f}$", transform=ax.transAxes, fontsize=10)

# Richness plots
# Similar pattern for richness vs R
ax = axs[1, 0]
ax.scatter(megaplot['R2_log10'], megaplot['spp_sqrt'], color='grey')
y_pred, sigma, model = gp_smooth(megaplot['R2_log10'].values, megaplot['spp_sqrt'].values, x_R, return_model=True)
ax.plot(x_R, y_pred, 'k--')
ax.fill_between(x_R, y_pred - 2*sigma, y_pred + 2*sigma, color='black', alpha=0.2)

# Add polynomial fit
x_poly, y_poly, _ = poly_fit(megaplot['R2_log10'], megaplot['spp_sqrt'])
ax.plot(x_poly, y_poly, 'b-')
ax.fill_between(x_poly, y_poly - 0.2, y_poly + 0.2, color='blue', alpha=0.2)

# Add subset model
y_pred_subset = gp_smooth(megaplotR['R2_log10'].values, megaplotR['spp_sqrt'].values, x_R2)[0]
ax.plot(x_R2, y_pred_subset, 'r-')
ax.fill_between(x_R2, y_pred_subset - 0.2, y_pred_subset + 0.2, color='red', alpha=0.2)

r2 = gam_r2(megaplot['R2_log10'].values, megaplot['spp_sqrt'].values, model)
ax.set_xlabel(r'$R^2 - 1$ (log scale)')
ax.set_ylabel('Richness')
ax.text(0.05, 0.95, 'd', transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top')
ax.text(0.7, 0.95, f"$r^2 = {r2:.3f}$", transform=ax.transAxes, fontsize=10)

# Richness vs D
ax = axs[1, 1]
ax.scatter(megaplot['D_theory'], megaplot['spp_sqrt'], color='grey')
y_pred, sigma, model = gp_smooth(megaplot['D_theory'].values, megaplot['spp_sqrt'].values, x_D, return_model=True)
ax.plot(x_D, y_pred, 'k--')
ax.fill_between(x_D, y_pred - 2*sigma, y_pred + 2*sigma, color='black', alpha=0.2)

# Add polynomial fit
x_poly, y_poly, _ = poly_fit(megaplot['D_theory'], megaplot['spp_sqrt'])
ax.plot(x_poly, y_poly, 'b-')
ax.fill_between(x_poly, y_poly - 0.2, y_poly + 0.2, color='blue', alpha=0.2)

r2 = gam_r2(megaplot['D_theory'].values, megaplot['spp_sqrt'].values, model)
ax.set_xlabel('D')
ax.set_ylabel('Richness')
ax.text(0.05, 0.95, 'e', transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top')
ax.text(0.7, 0.95, f"$r^2 = {r2:.3f}$", transform=ax.transAxes, fontsize=10)

# Richness vs H
ax = axs[1, 2]
ax.scatter(megaplot['HL0_log10'], megaplot['spp_sqrt'], color='grey')
y_pred, sigma, model = gp_smooth(megaplot['HL0_log10'].values, megaplot['spp_sqrt'].values, x_H, return_model=True)
ax.plot(x_H, y_pred, 'k--')
ax.fill_between(x_H, y_pred - 2*sigma, y_pred + 2*sigma, color='black', alpha=0.2)

# Add polynomial fit
x_poly, y_poly, _ = poly_fit(megaplot['HL0_log10'], megaplot['spp_sqrt'])
ax.plot(x_poly, y_poly, 'b-')
ax.fill_between(x_poly, y_poly - 0.2, y_poly + 0.2, color='blue', alpha=0.2)

r2 = gam_r2(megaplot['HL0_log10'].values, megaplot['spp_sqrt'].values, model)
ax.set_xlabel(r'$\Delta H / \sqrt{2} L_0$ (log scale)')
ax.set_ylabel('Richness')
ax.text(0.05, 0.95, 'f', transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top')
ax.text(0.7, 0.95, f"$r^2 = {r2:.3f}$", transform=ax.transAxes, fontsize=10)

# Abundance plots
# Abundance vs R
ax = axs[2, 0]
ax.scatter(megaplot['R2_log10'], megaplot['abd_sqrt'], color='grey')
y_pred, sigma, model = gp_smooth(megaplot['R2_log10'].values, megaplot['abd_sqrt'].values, x_R, return_model=True)
ax.plot(x_R, y_pred, 'k--')
ax.fill_between(x_R, y_pred - 2*sigma, y_pred + 2*sigma, color='black', alpha=0.2)

# Add polynomial fit
x_poly, y_poly, _ = poly_fit(megaplot['R2_log10'], megaplot['abd_sqrt'])
ax.plot(x_poly, y_poly, 'b-')
ax.fill_between(x_poly, y_poly - 0.5, y_poly + 0.5, color='blue', alpha=0.2)

# Add subset model
y_pred_subset = gp_smooth(megaplotR['R2_log10'].values, megaplotR['abd_sqrt'].values, x_R2)[0]
ax.plot(x_R2, y_pred_subset, 'r-')
ax.fill_between(x_R2, y_pred_subset - 0.5, y_pred_subset + 0.5, color='red', alpha=0.2)

r2 = gam_r2(megaplot['R2_log10'].values, megaplot['abd_sqrt'].values, model)
ax.set_xlabel(r'$R^2 - 1$ (log scale)')
ax.set_ylabel('Abundance')
ax.text(0.05, 0.95, 'g', transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top')
ax.text(0.7, 0.95, f"$r^2 = {r2:.3f}$", transform=ax.transAxes, fontsize=10)

# Abundance vs D
ax = axs[2, 1]
ax.scatter(megaplot['D_theory'], megaplot['abd_sqrt'], color='grey')
y_pred, sigma, model = gp_smooth(megaplot['D_theory'].values, megaplot['abd_sqrt'].values, x_D, return_model=True)
ax.plot(x_D, y_pred, 'k--')
ax.fill_between(x_D, y_pred - 2*sigma, y_pred + 2*sigma, color='black', alpha=0.2)

# Add polynomial fit
x_poly, y_poly, _ = poly_fit(megaplot['D_theory'], megaplot['abd_sqrt'])
ax.plot(x_poly, y_poly, 'b-')
ax.fill_between(x_poly, y_poly - 0.5, y_poly + 0.5, color='blue', alpha=0.2)

r2 = gam_r2(megaplot['D_theory'].values, megaplot['abd_sqrt'].values, model)
ax.set_xlabel('D')
ax.set_ylabel('Abundance')
ax.text(0.05, 0.95, 'h', transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top')
ax.text(0.7, 0.95, f"$r^2 = {r2:.3f}$", transform=ax.transAxes, fontsize=10)

# Abundance vs H
ax = axs[2, 2]
ax.scatter(megaplot['HL0_log10'], megaplot['abd_sqrt'], color='grey')
y_pred, sigma, model = gp_smooth(megaplot['HL0_log10'].values, megaplot['abd_sqrt'].values, x_H, return_model=True)
ax.plot(x_H, y_pred, 'k--')
ax.fill_between(x_H, y_pred - 2*sigma, y_pred + 2*sigma, color='black', alpha=0.2)

# Add polynomial fit
x_poly, y_poly, _ = poly_fit(megaplot['HL0_log10'], megaplot['abd_sqrt'])
ax.plot(x_poly, y_poly, 'b-')
ax.fill_between(x_poly, y_poly - 0.5, y_poly + 0.5, color='blue', alpha=0.2)

r2 = gam_r2(megaplot['HL0_log10'].values, megaplot['abd_sqrt'].values, model)
ax.set_xlabel(r'$\Delta H / \sqrt{2} L_0$ (log scale)')
ax.set_ylabel('Abundance')
ax.text(0.05, 0.95, 'i', transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top')
ax.text(0.7, 0.95, f"$r^2 = {r2:.3f}$", transform=ax.transAxes, fontsize=10)

plt.tight_layout()
plt.savefig("figs/figS7.png", dpi=300)
plt.close()

# Create a more detailed visualization for Fig 3 (site examples)
print("Creating Figure 3: Site examples...")
# First, create the main scatter plot with site-specific subsamples highlighted
fig = plt.figure(figsize=(8.8, 4))
gs = gridspec.GridSpec(3, 7, figure=fig)

# Create the main central scatter plot
ax_main = fig.add_subplot(gs[:, 2:5])
ax_main.scatter(dat['D_theory'], dat['R2_log10'], color='black', alpha=0.3)
ax_main.set_xlabel(r'Fractal dimension ($D$)')
ax_main.set_ylabel(r'Rugosity ($R^2 - 1$)')
ax_main.set_xlim(2, 2.6)
ax_main.set_ylim(-1.5, 0.7)
ax_main.set_yticks(np.log10([0.05, 0.1, 0.2, 0.5, 1, 2, 5]))
ax_main.set_yticklabels(['0.05', '0.1', '0.2', '0.5', '1', '2', '5'])

# Add text about height range
ax_main.text(2.09, 0.2, "Greater height\nrange", fontsize=7)
ax_main.text(2.53, -1.3, "Smaller height\nrange", fontsize=7)

# Highlight specific sites with different colors
site_list = ['Mermaid Cove', 'Osprey', 'Lagoon 2', 'Resort', 'South Island', 'Horseshoe']
colors = plt.cm.tab10(np.linspace(0, 1, len(site_list)))

# Try to load site images (if available)
site_images = {}
for site in site_list:
    dem_path = f"data/images_for_figures/{site.lower().replace(' ', '_')}_dem.jpg"
    mos_path = f"data/images_for_figures/{site.lower().replace(' ', '_')}_mos.jpg"
    
    try:
        site_images[f"{site}_dem"] = plt.imread(dem_path)
        site_images[f"{site}_mos"] = plt.imread(mos_path)
    except:
        print(f"Warning: Could not load images for site {site}")

# Create letter labels for sites
labels = ['a', 'b', 'c', 'd', 'e', 'f']

for i, site in enumerate(site_list):
    site_data = dat[dat['site'] == site]
    
    if len(site_data) >= 3:  # Need at least 3 points for convex hull
        # Get points for convex hull
        points = np.column_stack((site_data['D_theory'], site_data['R2_log10']))
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        hull_points = np.vstack([hull_points, hull_points[0]])  # Close the polygon
        
        # Plot hull and points
        ax_main.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.3, color=colors[i])
        ax_main.scatter(site_data['D_theory'], site_data['R2_log10'], color=colors[i], s=20)
        
        # Find center of points for label
        center_x = np.mean(site_data['D_theory'])
        center_y = np.mean(site_data['R2_log10'])
        
        # Add site label
        ax_main.text(center_x, center_y, labels[i], fontsize=12)
    
    # Create site image subplots
    if i < 3:  # First three sites go on the left
        row = i
        col_dem = 0
        col_mos = 1
    else:  # Last three sites go on the right
        row = i - 3
        col_dem = 5
        col_mos = 6
    
    # Add DEM image subplot
    ax_dem = fig.add_subplot(gs[row, col_dem])
    ax_dem.set_xticks([])
    ax_dem.set_yticks([])
    ax_dem.spines['top'].set_visible(False)
    ax_dem.spines['right'].set_visible(False)
    ax_dem.spines['bottom'].set_visible(False)
    ax_dem.spines['left'].set_visible(False)
    
    # Add mosaic image subplot
    ax_mos = fig.add_subplot(gs[row, col_mos])
    ax_mos.set_xticks([])
    ax_mos.set_yticks([])
    ax_mos.spines['top'].set_visible(False)
    ax_mos.spines['right'].set_visible(False)
    ax_mos.spines['bottom'].set_visible(False)
    ax_mos.spines['left'].set_visible(False)
    
    # Display images if available
    if f"{site}_dem" in site_images:
        ax_dem.imshow(site_images[f"{site}_dem"])
        ax_dem.text(0.1, 0.9, labels[i], transform=ax_dem.transAxes, fontsize=12, 
                   fontweight='bold', color='black')
    else:
        ax_dem.text(0.5, 0.5, f"{site}\nDEM", ha='center', va='center')
        
    if f"{site}_mos" in site_images:
        ax_mos.imshow(site_images[f"{site}_mos"])
    else:
        ax_mos.text(0.5, 0.5, f"{site}\nPhoto", ha='center', va='center')

plt.tight_layout()
plt.savefig("figs/fig3.pdf", dpi=300)
plt.close()

# Fig 4: Geometric/biodiversity coupling
print("Creating Figure 4: Geometric/biodiversity coupling...")
fig = plt.figure(figsize=(6, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], figure=fig)

# Generate grid for contour plot
D_grid = np.linspace(min(megaplot['D_theory']), max(megaplot['D_theory']), 50)
R_grid = np.linspace(min(megaplot['R2_log10']), max(megaplot['R2_log10']), 50)
D_mesh, R_mesh = np.meshgrid(D_grid, R_grid)

# Calculate H_mesh for each D,R combination using the HL0_func
H_mesh = np.zeros_like(D_mesh)
for i in range(D_mesh.shape[0]):
    for j in range(D_mesh.shape[1]):
        try:
            H_mesh[i, j] = HL0_func(D_mesh[i, j], np.sqrt(10**R_mesh[i, j] + 1), L, L0)
        except:
            H_mesh[i, j] = np.nan

# Create training data for PIE prediction
X_train = np.column_stack([
    megaplot['D_theory'].values,
    megaplot['R2_log10'].values,
    megaplot['HL0_log10'].values
])
y_train = megaplot['pie_asin'].values

# Set up kernel for 3D GPR
kernel = 1.0 * RBF(length_scale=[0.1, 0.1, 0.1]) + WhiteKernel(noise_level=0.1)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=5)
gpr.fit(X_train, y_train)

# Prepare grid for prediction
X_pred = np.column_stack([
    D_mesh.ravel(),
    R_mesh.ravel(),
    np.log10(H_mesh).ravel()
])
# Remove NaN rows
nan_mask = ~np.isnan(X_pred).any(axis=1)
X_pred_clean = X_pred[nan_mask]

# Make predictions
if len(X_pred_clean) > 0:
    Z_pred = np.full(D_mesh.size, np.nan)
    Z_pred[nan_mask] = gpr.predict(X_pred_clean)
    Z_pred = Z_pred.reshape(D_mesh.shape)
else:
    Z_pred = np.full_like(D_mesh, np.nan)

# Create points for convex hull of data to mask predictions outside observed range
hull_points = np.column_stack([megaplot['D_theory'], megaplot['R2_log10']])
hull = ConvexHull(hull_points)

# Function to check if points are in hull
def in_hull(points, hull):
    """Check if points are inside the convex hull"""
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull.points[hull.vertices])
    return hull.find_simplex(points) >= 0

# Apply hull mask to predictions
grid_points = np.column_stack([D_mesh.ravel(), R_mesh.ravel()])
in_hull_mask = in_hull(grid_points, hull).reshape(D_mesh.shape)
Z_pred_masked = np.where(in_hull_mask, Z_pred, np.nan)

# Upper plot: PIE contour plot
ax1 = fig.add_subplot(gs[0])
contourf = ax1.contourf(D_mesh, R_mesh, np.sin(Z_pred_masked), 
                        levels=np.linspace(0, 1, 11), cmap='Reds_r', alpha=0.7)
contour = ax1.contour(D_mesh, R_mesh, np.sin(Z_pred_masked), 
                     levels=np.linspace(0, 1, 11), colors='black', linewidths=0.5)
scatter = ax1.scatter(megaplot['D_theory'], megaplot['R2_log10'], 
                     color='black', alpha=0.3, s=15)

ax1.set_xlabel('$D$')
ax1.set_ylabel(r'Rugosity ($R^2 - 1$)')
ax1.set_xlim(2, 2.6)
ax1.set_ylim(-1.7, 0.7)
ax1.set_yticks(np.log10([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]))
ax1.set_yticklabels(['0.02', '0.05', '0.1', '0.2', '0.5', '1', '2', '5'])
ax1.text(0.05, 0.95, 'a', transform=ax1.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top')
ax1.set_title("Diversity (probability of interspecific encounter)")

# Add additional text and annotations
ax1.text(2.09, 0.2, "Greater height\nrange", fontsize=7)
ax1.text(2.53, -1.3, "Smaller height\nrange", fontsize=7)

# Add a colorbar
cbar = plt.colorbar(contourf, ax=ax1, shrink=0.8)
cbar.set_label('Probability of interspecific encounter')

# R^2 value
try:
    # Calculate R^2 for the PIE prediction model
    y_pred = gpr.predict(X_train)
    ss_tot = np.sum((y_train - np.mean(y_train))**2)
    ss_res = np.sum((y_train - y_pred)**2)
    r_squared = 1 - (ss_res / ss_tot)
    ax1.text(0.7, 0.95, f"$r^2 = {r_squared:.3f}$", transform=ax1.transAxes, fontsize=10)
except:
    print("Warning: Could not calculate R^2 for PIE model")

# Lower plot: Trimodal DEM image
ax2 = fig.add_subplot(gs[1])
# Try to load the Trimodal DEM image
try:
    trimodal_img = plt.imread("data/images_for_figures/Trimodal_dem4a.jpg")
    ax2.imshow(trimodal_img)
except:
    ax2.text(0.5, 0.5, "Trimodal DEM image would go here", 
            ha='center', va='center', fontsize=12)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.text(0.05, 0.95, 'b', transform=ax2.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top')

plt.tight_layout()
plt.savefig("figs/fig4.pdf", dpi=300)
plt.close()

# Additional diagnostic plots

# Fig S5: Fractal dimension residuals
print("Creating Supplementary Figures...")
plt.figure(figsize=(4, 4))
residuals = dat['D_theory'] - dat['D']
plt.hist(residuals, bins=50, density=True, alpha=0.7)
plt.xlabel('D theory - D empirical')
plt.axvline(x=0, color='red', linestyle='-', linewidth=2)

# Add a density curve
from scipy.stats import gaussian_kde
density = gaussian_kde(residuals)
x_vals = np.linspace(min(residuals), max(residuals), 100)
plt.plot(x_vals, density(x_vals), 'b-', linewidth=2)

plt.tight_layout()
plt.savefig("figs/figS5.png", dpi=300)
plt.close()

# Fig S6: Abundance versus richness
plt.figure(figsize=(5, 5))
plt.scatter(megaplot['abd'], megaplot['spp'], color='grey')
plt.xlabel('Abundance')
plt.ylabel('Richness')

# Fit a model to the sqrt-transformed data
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(
    megaplot['abd_sqrt'], megaplot['spp_sqrt'])

# Create prediction line
x_new = np.linspace(min(megaplot['abd_sqrt']), max(megaplot['abd_sqrt']), 100)
y_pred = intercept + slope * x_new
y_lower = y_pred - 1.96 * std_err
y_upper = y_pred + 1.96 * std_err

# Convert back to original scale for plotting
plt.fill_between(x_new**2, y_lower**2, y_upper**2, alpha=0.2, color='grey')
plt.plot(x_new**2, y_pred**2, 'k--')

plt.tight_layout()
plt.savefig("figs/figS6.png", dpi=300)
plt.close()

# Diagnostic plots for GAMs
plt.figure(figsize=(8, 12))
fig, axs = plt.subplots(4, 3, figsize=(12, 16))

# Helper function for model diagnostics
def plot_diagnostics(ax, x, y, y_pred, title):
    """Plot diagnostic plots for GAM models"""
    residuals = y - y_pred
    
    # Residuals vs fitted
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')
    ax.set_title(title)
    
    # Add lowess trend line if scipy.stats has it
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        lowess_y = lowess(residuals, y_pred, frac=0.6)[:, 1]
        ax.plot(y_pred, lowess_y, 'r-', linewidth=1)
    except:
        pass

# Create diagnostic plots for richness models
models = []
titles = ['Richness ~ R', 'Richness ~ D', 'Richness ~ H']

# Get predictions for each model
y_pred_R, _, model_R = gp_smooth(megaplot['R2_log10'].values, megaplot['spp_sqrt'].values, 
                               megaplot['R2_log10'].values, return_model=True)
y_pred_D, _, model_D = gp_smooth(megaplot['D_theory'].values, megaplot['spp_sqrt'].values, 
                               megaplot['D_theory'].values, return_model=True)
y_pred_H, _, model_H = gp_smooth(megaplot['HL0_log10'].values, megaplot['spp_sqrt'].values, 
                               megaplot['HL0_log10'].values, return_model=True)

models = [model_R, model_D, model_H]
x_vars = [megaplot['R2_log10'].values, megaplot['D_theory'].values, megaplot['HL0_log10'].values]
y_preds = [y_pred_R, y_pred_D, y_pred_H]

# Plot diagnostics for each richness model
for i, (model, x, y_pred, title) in enumerate(zip(models, x_vars, y_preds, titles)):
    plot_diagnostics(axs[0, i], x, megaplot['spp_sqrt'].values, y_pred, title)

# Similar for PIE models
titles = ['PIE ~ R', 'PIE ~ D', 'PIE ~ H']

# Get predictions for each model
y_pred_R, _, model_R = gp_smooth(megaplot['R2_log10'].values, megaplot['pie_asin'].values, 
                               megaplot['R2_log10'].values, return_model=True)
y_pred_D, _, model_D = gp_smooth(megaplot['D_theory'].values, megaplot['pie_asin'].values, 
                               megaplot['D_theory'].values, return_model=True)
y_pred_H, _, model_H = gp_smooth(megaplot['HL0_log10'].values, megaplot['pie_asin'].values, 
                               megaplot['HL0_log10'].values, return_model=True)

models = [model_R, model_D, model_H]
x_vars = [megaplot['R2_log10'].values, megaplot['D_theory'].values, megaplot['HL0_log10'].values]
y_preds = [y_pred_R, y_pred_D, y_pred_H]

# Plot diagnostics for each PIE model
for i, (model, x, y_pred, title) in enumerate(zip(models, x_vars, y_preds, titles)):
    plot_diagnostics(axs[1, i], x, megaplot['pie_asin'].values, y_pred, title)

# Similar for abundance models
titles = ['Abundance ~ R', 'Abundance ~ D', 'Abundance ~ H']

# Get predictions for each model
y_pred_R, _, model_R = gp_smooth(megaplot['R2_log10'].values, megaplot['abd_sqrt'].values, 
                               megaplot['R2_log10'].values, return_model=True)
y_pred_D, _, model_D = gp_smooth(megaplot['D_theory'].values, megaplot['abd_sqrt'].values, 
                               megaplot['D_theory'].values, return_model=True)
y_pred_H, _, model_H = gp_smooth(megaplot['HL0_log10'].values, megaplot['abd_sqrt'].values, 
                               megaplot['HL0_log10'].values, return_model=True)

models = [model_R, model_D, model_H]
x_vars = [megaplot['R2_log10'].values, megaplot['D_theory'].values, megaplot['HL0_log10'].values]
y_preds = [y_pred_R, y_pred_D, y_pred_H]

# Plot diagnostics for each abundance model
for i, (model, x, y_pred, title) in enumerate(zip(models, x_vars, y_preds, titles)):
    plot_diagnostics(axs[2, i], x, megaplot['abd_sqrt'].values, y_pred, title)

# QQ plots for residuals
for i, y_pred in enumerate(y_preds):
    residuals = megaplot['abd_sqrt'].values - y_pred
    stats.probplot(residuals, dist="norm", plot=axs[3, i])
    axs[3, i].set_title(f"QQ Plot - {titles[i]}")

plt.tight_layout()
plt.savefig("figs/figS8_diag.png", dpi=300)
plt.close()

print("Analysis complete! All figures saved to the 'figs' directory.")