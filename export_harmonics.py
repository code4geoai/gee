"""
export_harmonics.py - Google Earth Engine NDVI Time Series Analysis with Harmonic Regression
Saved on: [Today's Date]
Key Features:
1. Sentinel-2 cloud-masked NDVI time series
2. Dynamic World land cover integration
3. Harmonic trend analysis
4. Phase/Amplitude visualization
5. Automated export to Google Drive
"""

import ee
import math
import datetime

# Initialize Earth Engine
try:
    ee.Initialize()
    print("Earth Engine initialized successfully")
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Configuration Class
class Config:
    # Time Parameters
    START_DATE = '2022-10-01'
    END_DATE = '2023-06-02'
    
    # Cloud Masking Parameters
    CLOUD_FILTER = 60  # (%) Maximum cloud cover percentage
    CLD_PRB_THRESH = 50  # (%) Cloud probability threshold
    NIR_DRK_THRESH = 0.15  # NIR dark pixel threshold
    CLD_PRJ_DIST = 1  # km (cloud projection distance)
    BUFFER = 50  # m (dilation buffer for cloud masks)
    
    # Area Parameters
    AOI_COUNTRY_CODE = 256  # FAO country code
    CRS = 'EPSG:27700'  # Coordinate Reference System
    
    # Export Parameters
    EXPORT_FOLDER = 'GEE_Exports'
    EXPORT_PREFIX = 'ndvi_harmonics_'
    EXPORT_SCALE = 10  # meters

cfg = Config()

# Helper Functions
def print_info(collection, name):
    """Print basic collection information"""
    size = collection.size().getInfo()
    date_range = ee.Image(collection.first()).date().format('YYYY-MM-dd').getInfo()
    print(f"{name}: {size} images from {date_range}")

# Main Processing Functions
def get_s2_collection(aoi, start_date, end_date):
    """Get cloud-masked Sentinel-2 collection"""
    s2_sr = (ee.ImageCollection('COPERNICUS/S2_SR')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cfg.CLOUD_FILTER)))
    
    s2_cloudless = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))
    
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(
        primary=s2_sr,
        secondary=s2_cloudless,
        condition=ee.Filter.equals(
            leftField='system:index',
            rightField='system:index'
        )
    ))

def add_cloud_shadow_mask(img):
    """Add cloud and shadow masks to an image"""
    # Cloud detection
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
    is_cloud = cld_prb.gt(cfg.CLD_PRB_THRESH).rename('clouds')
    
    # Shadow detection
    not_water = img.select('SCL').neq(6)
    dark_pixels = img.select('B8').lt(cfg.NIR_DRK_THRESH*1e4).multiply(not_water)
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
    
    cld_proj = (img.select('clouds')
        .directionalDistanceTransform(shadow_azimuth, cfg.CLD_PRJ_DIST*10)
        .reproject(crs=img.select(0).projection(), scale=100)
        .select('distance')
        .mask()
        .rename('cloud_transform'))
    
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')
    
    # Combined mask
    is_cld_shdw = (img.addBands(is_cloud)
        .addBands(shadows)
        .select('clouds').add(shadows).gt(0)
    
    return img.addBands(is_cld_shdw.rename('cloudmask'))

def apply_mask(img):
    """Apply cloud/shadow mask to reflectance bands"""
    return img.select('B.*').updateMask(img.select('cloudmask').Not())

def add_harmonic_vars(img):
    """Add time and harmonic variables for regression"""
    date = ee.Date(img.get('system:time_start'))
    years = date.difference(ee.Date('1970-01-01'), 'year')
    time_rad = years.multiply(2 * math.pi)
    
    return (img
        .addBands(img.normalizedDifference(['B8', 'B4']).rename('NDVI'))
        .addBands(ee.Image(years).rename('t'))
        .addBands(time_rad.cos().rename('cos'))
        .addBands(time_rad.sin().rename('sin'))
        .addBands(ee.Image.constant(1).rename('constant')))

# Main Processing Chain
def main():
    # 1. Set up Area of Interest
    aoi = ee.FeatureCollection('FAO/GAUL/2015/level0') \
            .filterMetadata('ADM0_CODE', 'equals', cfg.AOI_COUNTRY_CODE)
    print(f"AOI: {aoi.first().get('ADM0_NAME').getInfo()}")

    # 2. Get cloud-masked Sentinel-2 collection
    s2_col = get_s2_collection(aoi, cfg.START_DATE, cfg.END_DATE)
    s2_masked = s2_col.map(add_cloud_shadow_mask).map(apply_mask)
    print_info(s2_masked, "Cloud-masked Sentinel-2")

    # 3. Add harmonic regression variables
    harmonic_col = s2_masked.map(add_harmonic_vars)
    
    # 4. Perform harmonic regression
    harmonic_vars = ee.List(['constant', 't', 'cos', 'sin'])
    harmonic_trend = harmonic_col.select(harmonic_vars.add('NDVI')) \
        .reduce(ee.Reducer.linearRegression(harmonic_vars.length(), 1))
    
    # 5. Compute phase and amplitude
    coeffs = harmonic_trend.select('coefficients') \
        .arrayProject([0]) \
        .arrayFlatten([harmonic_vars])
    
    phase = coeffs.select('cos').atan2(coeffs.select('sin'))
    amplitude = coeffs.select('cos').hypot(coeffs.select('sin'))
    
    # 6. Create visualization
    rgb = (phase.unitScale(-math.pi, math.pi)
        .addBands(amplitude.multiply(2.5))
        .addBands(ee.Image(1))
        .hsvToRgb()
        .multiply(255)
        .toByte()
    
    # 7. Export results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    task = ee.batch.Export.image.toDrive(
        image=rgb.clip(aoi),
        description='NDVI_Harmonics',
        folder=cfg.EXPORT_FOLDER,
        fileNamePrefix=f"{cfg.EXPORT_PREFIX}{timestamp}",
        region=aoi.geometry(),
        scale=cfg.EXPORT_SCALE,
        crs=cfg.CRS,
        maxPixels=1e13
    )
    task.start()
    print(f"Export started with Task ID: {task.id}")
    print(f"Monitor at: https://code.earthengine.google.com/tasks")

if __name__ == "__main__":
    main()