pip index versions triton -i https://repos.metax-tech.com/r/maca-pypi/simple --trusted-host repos.metax-tech.com
pip index versions flash-attn \
-i https://repos.metax-tech.com/r/maca-pypi/simple \
--trusted-host repos.metax-tech.com

pip install "torch==2.8.0+metax3.3.0.2" -i https://repos.metax-tech.com/r/maca-pypi/simple --trusted-host repos.metax-tech.com

pip index versions tensorflow -i https://repos.metax-tech.com/r/maca-pypi/simple --trusted-host repos.metax-tech.com