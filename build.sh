ROOT=$(pwd)
echo "ROOT: $ROOT"
chmod 777 /opt/conda/envs/catgrasp/lib/python3.7/site-packages/cmake* -R
cd $ROOT/PointGroup && bash build.sh
cd $ROOT/ikfast_pybind && rm -rf build *egg* && python setup.py develop
chmod 777 /opt/conda/envs/catgrasp/lib/python3.7/site-packages/cmake* -R
cd $ROOT/my_cpp && rm -rf build *.so *egg* && python setup.py develop
