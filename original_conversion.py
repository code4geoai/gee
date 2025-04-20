import ee
import math

# Initialize Earth Engine
ee.Initialize()

# Constants
START_DATE = '2022-10-01'
END_DATE = '2023-06-02'
CLOUD_FILTER = 60
CLD_PRB_THRESH = 50
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50

# Area of Interest
aoi = ee.FeatureCollection('FAO/GAUL/2015/level0').filterMetadata('ADM0_CODE', 'equals', 256)
geometry = aoi.geometry()

# Dynamic World Processing
dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
        .filterDate(START_DATE, END_DATE) \
        .filterBounds(aoi)
classification = dw.select('label')
dwComposite = classification.reduce(ee.Reducer.mode())

croplands = dwComposite.updateMask(dwComposite.clip(aoi).eq(4))
grass = dwComposite.updateMask(dwComposite.clip(aoi).eq(2))
combinedMask = croplands.blend(grass)

def get_s2_sr_cld_col(aoi, start_date, end_date):
    s2_sr_col = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER))

    s2_cloudless_col = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date)

    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(
        primary=s2_sr_col,
        secondary=s2_cloudless_col,
        condition=ee.Filter.equals(
            leftField='system:index',
            rightField='system:index'
        )
    ))

def add_cloud_bands(img):
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')
    return img.addBands(ee.Image([cld_prb, is_cloud]))

def add_shadow_bands(img):
    not_water = img.select('SCL').neq(6)
    SR_BAND_SCALE = 1e4
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
    
    cld_proj = img.select('clouds') \
        .directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10) \
        .reproject(crs=img.select(0).projection(), scale=100) \
        .select('distance') \
        .mask() \
        .rename('cloud_transform')
    
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

def add_cld_shdw_mask(img):
    img_cloud = add_cloud_bands(img)
    img_cloud_shadow = add_shadow_bands(img_cloud)
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)
    
    is_cld_shdw = is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20) \
        .reproject(crs=img.select([0]).projection(), scale=20) \
        .rename('cloudmask')
    
    return img_cloud_shadow.addBands(is_cld_shdw)

def apply_cld_shdw_mask(img):
    # Corrected .not() syntax here
    not_cld_shdw = img.select('cloudmask').Not()
    return img.select('B.*').updateMask(not_cld_shdw)

# Main processing
s2_sr_cld_col_eval = get_s2_sr_cld_col(aoi, START_DATE, END_DATE)
l8toa = s2_sr_cld_col_eval.map(add_cld_shdw_mask).map(apply_cld_shdw_mask)

# Time series analysis
timeField = 'system:time_start'

def add_variables(image):
    date = ee.Date(image.get(timeField))
    years = date.difference(ee.Date('1970-01-01'), 'year')
    return image \
        .addBands(image.normalizedDifference(['B8', 'B4']).rename('NDVI')) \
        .addBands(ee.Image(years).rename('t')) \
        .addBands(ee.Image.constant(1))

filteredLandsat = l8toa.filterBounds(geometry).map(add_variables)

# Linear trend
independents = ee.List(['constant', 't'])
dependent = ee.String('NDVI')
trend = filteredLandsat.select(independents.add(dependent)) \
    .reduce(ee.Reducer.linearRegression(independents.length(), 1))
coefficients = trend.select('coefficients') \
    .arrayProject([0]) \
    .arrayFlatten([independents])

def detrend_func(image):
    return image.select(dependent) \
        .subtract(image.select(independents).multiply(coefficients).reduce('sum')) \
        .rename(dependent) \
        .copyProperties(image, [timeField])

detrended = filteredLandsat.map(detrend_func)

# Export
eightbitRGB = combinedMask.visualize(min=0, max=1, palette=['000000', 'FFFFFF'])

task = ee.batch.Export.image.toDrive(
    image=eightbitRGB.clip(aoi),
    description='NDVI_Analysis',
    folder='GEE_Exports',
    fileNamePrefix='ndvi_analysis',
    region=aoi.geometry(),
    scale=10,
    crs='EPSG:27700',
    maxPixels=1e13
)
task.start()

print(f"Export started with task ID: {task.id}")