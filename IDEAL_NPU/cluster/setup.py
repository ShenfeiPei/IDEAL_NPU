import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('cluster', parent_package, top_path)
    config.add_subpackage('AGCI')
    config.add_subpackage('DNC')
    config.add_subpackage('EDG')
    config.add_subpackage('EEBC')
    config.add_subpackage('FCDMF')
    config.add_subpackage('FINCH')
    config.add_subpackage('FSCAG')
    config.add_subpackage('PCN')
    config.add_subpackage('SC')
    config.add_subpackage('SNNDPC')
    config.add_subpackage('SSC')
    config.add_subpackage('WBKM')
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
