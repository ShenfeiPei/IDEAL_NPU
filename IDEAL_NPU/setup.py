import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config = Configuration('IDEAL_NPU', parent_package, top_path)
    config.add_subpackage('cluster')
    config.add_data_dir('dataset')
    config.add_data_dir('demo')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
