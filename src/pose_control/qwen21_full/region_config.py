"""Versioned, stdlib-only regional FM settings; legacy records stay unchanged."""
FAMILIES = {'person': 2, 'joint': 2, 'hands': 4, 'background': 1}
MASS_FLOORS = {'person': 4., 'joint': 1., 'hands': .5, 'background': 16.}
PRESETS = {
    'uniform': {'global': 1., 'person': 0., 'joint': 0., 'hands': 0., 'background': 0.},
    'region_balanced_v1': {'global': .1, 'person': .3, 'joint': .15, 'hands': .35, 'background': .1},
}


def validate_region_mode(loss):
    mode = loss.region_mode
    if mode not in ('legacy', *PRESETS):
        raise ValueError('Unknown regional FM mode')
    if mode != 'legacy' and (any(loss.spatial_coefficients) or loss.face or loss.hand):
        raise ValueError('Regional FM replaces legacy spatial/Face/Hand velocity terms')


def compatible_record(value):
    """Omit only new inactive defaults, retaining every historical field."""
    for route in value.get('stage', {}).get('routes', ()):
        loss = route.get('loss', {})
        if loss.get('region_mode') == 'legacy':
            loss.pop('region_mode')
        # The bg035 runtime snapshot introduced this optional default separately.
        if loss.get('spatial_bg') == 1.:
            loss.pop('spatial_bg')
    for source in value.get('sources', ()):
        if source.get('region_supervision') is None:
            source.pop('region_supervision', None)
    return value
