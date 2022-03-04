set -e

cd lib/spconv
rm -rf build dist
python setup.py bdist_wheel
cd dist
pip install $(ls|grep whl) --force-reinstall

cd ../..
cd pointgroup_ops
rm -rf build PG_OP.egg-info *.so
python setup.py develop