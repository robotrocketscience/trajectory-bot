***Utilizing Reinforcement Learning for Interplanetary Trajectory
Optimization***

***Final Project ASTE 581***

***robotrocketscience 5/3/2021***

Reinforcement learning for trajectory optimization has great potential
for mission planning. I zealously made sure to follow the progress
report as well as many other sources on Proximal Policy Optimization.
However, due to the complexity of the project, and the time constraint
imposed, it appears I have bitten off much more than I could chew.

The three main Python files used in this project are mainPPO.py,
ppo_torch.py, and basic_env.py.

basic_env.py contains the environment class as well as a number of
helper functions. This file defines the environment experienced and
acted upon by the RL agent. Essentially, basic_env takes in actions and
outputs a state and reward.

ppo_torch.py contains the Agent class, composed of an actor and critic
represented by linear neural networks. Essentially, ppo_torch.py takes
as input state observations and reward, and outputs an action. The
neural network weights are then updated based on the reward associated
with the observation.

MainPPO.py is the main function, initializing the Agent class and
loading the environment, then running a user-selected number of training
sessions or “games.” mainPPO.py also plots the training progress of the
agent, as shown here:

<img src="media/image1.png" style="width:5.99931in;height:3.99931in" />

The good news is: I was successful in implementing a reinforcement
learning algorithm utilizing the fundamental physics of orbital
mechanics. The bad news is: the agent was unable to learn an optimal,
let alone sub-optimal trajectory. While the agent is able to improve its
score after running many games, it does not ever achieve the desired
target orbit conditions. No matter what reward function I used, I was
unable to see a decrease in the distance between the spacecraft and the
target planet. This failure is due to several factors, which I will
outline as follows:

**Unfamiliarity with Python**

I decided to use Python because of the broad support for Reinforcement
Learning packages, such as Torch and Gym. Unfortunately, learning the
Python knowledge needed to implement this program was a monumental
project in itself. I am grateful I had the opportunity to learn Python
in-depth, as I did not know nearly enough to successfully complete this
project. I am immensely proud of the fact that I was able to implement a
RL agent to learn a trajectory at all.

While I was able to perform hundreds if not thousands of training runs,
I was unable to figure out how to load the previously trained neural
network weights in order to continue training. This meant every time I
ran the algorithm to train, I was essentially starting from scratch.

**Too Many Variables**

When setting up the problem, I decided to use the observation space
composed of x,y,z positions of the major bodies, the x,y,z position and
velocity of the spacecraft, and the orientation of the spacecraft, as
well as the fuel. Then, to complicate matters, the spacecraft could
choose from 8 distinct actions, 6 of which involve changing the
orientation of the spacecraft. Many training sessions were spent with
the spacecraft constantly flipping around in space without advancing in
any meaningful way. To counteract this, I included a punishment for
changing orientation in the reward function, as well as making an
orientation change cost fuel. This helped reduce the number of
orientation changes.

In retrospect, I should have focused on making this problem 2
dimensional rather than stubbornly using 3 dimensional space. A 2
dimensional space would make only one orientation change (yawing)
relevant, and would reduce the number of variables and therefore
observations.

**Too Long to Train**

Initially, I had planned an Earth-Mars trajectory. However, I quickly
figured out that even if each training step took half a second to
complete, I would be looking at days of training just to complete one
episode. In order to combat this, I changed the target body to the Moon.
This made the fuel requirement much smaller, meaning each training
episode would only take between 30-80 seconds. Still, this means
training overnight (8 hours) resulted in only 400-800 or so episodes.

**The Reward Function**

I severely underestimated the reward function. I tried many different
reward functions, some did nothing, others improved performance
somewhat. Eventually I settled on a number of rewards based on a
gradient which increases as the spacecraft approaches the desired orbit.
I also included a massive punishment for impacting the origin or target
body, as well as a punishment for using too much fuel and changing
orientation too much. Finally, using the definition of a circular orbit,
I provided a massive bonus reward for achieving any of three conditions
for a lunar orbit: orbit period, orbit energy, and circular velocity.
These values are outlined in the function “getReward().”

Unfortunately, none of these reward functions helped. The distance to
the target body steadily increases rather than decreases no matter what
I tried. As I ran out of time to complete this project, I thought about
a reward function which I believe would provide the desired behavior:

A good reward function might be to establish an ellipse with the origin
body and target body coordinates as it’s foci. Then, points along that
ellipse can be generated every so often (say, every 10km), and the
spacecraft path can be rewarded (or punished) depending on its deviation
from those points. However, this approach begs the question: if such an
orbital path can be defined in the first place, what use is it to
utilize a RL agent to follow the path? The entire point is to see if the
optimum path can be learned without such direct supervision and
guidance. Alas, I ran out of time to implement any meaningful or
successful reward function.

**Future Work**

Given more time to work on this project, I would definitely explore
other reward functions to see if I can converge on an actually useful
and successful one. I would also split the orientation and trajectory
planning into two different agents. One agent would be responsible for
pointing the spacecraft in the correct orientation, while the other
agent can focus on optimizing the trajectory.

Another avenue to explore would be multi-agent learning. The modules I
used for this project support using multiple agents to simultaneously
learn, using two agents per CPU/GPU core. This would have been useful in
speeding up the learning process. This, of course, assumes the reward
function is correctly defined.

As mentioned earlier, a major pitfall was an inability to implement
model loading and saving correctly. I would want to be able to load the
previously used neural network weights each time I run the program, so
that minor adjustments could be made without starting from absolute
zero.

**Conclusion**

I am extremely humbled by my lack of success in doing this project.
While Reinforcement Learning can be a powerful tool to accomplish this
task, it appears that my skill and knowledge were insufficient in
successfully completing it. However, I do take pride in how much I
learned and was able to achieve despite the lack of mission success.
After all, the program does run, the agent does prioritize thrusting and
coasting over irrelevant orientation changes, and the program is
converging on an optimum, however irrelevant that optimum is. This gives
me hope that with a proper reward function, the agent can in fact be
trained to achieve the desired behavior.

Despite unsuccessfully programming a RL agent to optimize interplanetary
trajectories, I learned a lot about modeling orbital mechanics, Python
programming, and Reinforcement Learning as a technology. I intend on
taking ASTE 583 next semester, and hope I will be able to work on this
particular project more. I am still optimistic about the possibility and
benefit of using RL to accomplish mission planning tasks.

**Appendix**

In order to run this code, use Python to set up a Conda environment as
follows:

Install anaconda3

conda create -n yourenvname python=3.8

conda activate yourenvname

type “conda list” and ensure all the following packages are installed:

**\# packages in environment at /home/yoshi/anaconda3/envs/MLpy:**

**\#**

|                                                       |
|:------------------------------------------------------|
| **\# Name Version Build Channel**                     |
| **\_libgcc_mutex 0.1 main**                           |
| **alabaster 0.7.12 pyhd3eb1b0_0**                     |
| **anyio 2.1.0 py38h578d9bd_0 conda-forge**            |
| **appdirs 1.4.4 py_0**                                |
| **argh 0.26.2 py38_0**                                |
| **argon2-cffi 20.1.0 py38h25fe258_2 conda-forge**     |
| **astroid 2.5 py38h06a4308_1**                        |
| **astropy 3.2.3 py38h516909a_0 conda-forge**          |
| **astroquery 0.4.1 pyh9f0ad1d_0 conda-forge**         |
| **async_generator 1.10 pyhd3eb1b0_0**                 |
| **atomicwrites 1.4.0 py_0**                           |
| **attrs 20.3.0 pyhd3eb1b0_0**                         |
| **autopep8 1.5.5 pyhd3eb1b0_0**                       |
| **babel 2.9.0 pyhd3eb1b0_0**                          |
| **backcall 0.2.0 pyhd3eb1b0_0**                       |
| **beautifulsoup4 4.9.3 pyhb0f4dca_0 conda-forge**     |
| **black 19.10b0 py_0**                                |
| **blas 1.0 mkl**                                      |
| **bleach 3.3.0 pyhd3eb1b0_0**                         |
| **box2d-py 2.3.8 py38h950e882_2 conda-forge**         |
| **brotlipy 0.7.0 py38h27cfd23_1003**                  |
| **bzip2 1.0.8 h516909a_3 conda-forge**                |
| **ca-certificates 2021.4.13 h06a4308_1**              |
| **certifi 2020.12.5 py38h578d9bd_1 conda-forge**      |
| **cffi 1.14.5 py38h261ae71_0**                        |
| **chardet 4.0.0 py38h06a4308_1003**                   |
| **click 7.1.2 pyhd3eb1b0_0**                          |
| **cloudpickle 1.6.0 py_0**                            |
| **colorama 0.4.4 pyhd3eb1b0_0**                       |
| **cryptography 3.3.1 py38h3c74f83_1**                 |
| **cudatoolkit 10.2.89 hfd86e86_1**                    |
| **cycler 0.10.0 py38_0**                              |
| **dbus 1.13.18 hb2f20db_0**                           |
| **decorator 4.4.2 pyhd3eb1b0_0**                      |
| **defusedxml 0.6.0 pyhd3eb1b0_0**                     |
| **diff-match-patch 20200713 py_0**                    |
| **docutils 0.16 py38_1**                              |
| **entrypoints 0.3 py38_0**                            |
| **expat 2.2.10 he6710b0_2**                           |
| **ffmpeg 4.3.1 h3215721_1 conda-forge**               |
| **flake8 3.8.4 py_0**                                 |
| **fontconfig 2.13.1 h6c09931_0**                      |
| **freetype 2.10.4 h5ab3b9f_0**                        |
| **future 0.18.2 py38_1**                              |
| **glib 2.67.4 h36276a3_1**                            |
| **gmp 6.2.1 h58526e2_0 conda-forge**                  |
| **gnutls 3.6.13 h85f3911_1 conda-forge**              |
| **gst-plugins-base 1.14.0 h8213a91_2**                |
| **gstreamer 1.14.0 h28cd5cc_2**                       |
| **gym 0.18.0 py38h81c977d_0 conda-forge**             |
| **gym-push 0.0.1 dev_0 \<develop\>**                  |
| **html5lib 1.1 pyh9f0ad1d_0 conda-forge**             |
| **hypothesis 6.10.0 pyhd8ed1ab_0 conda-forge**        |
| **icu 58.2 he6710b0_3**                               |
| **idna 2.10 pyhd3eb1b0_0**                            |
| **imagesize 1.2.0 pyhd3eb1b0_0**                      |
| **importlib-metadata 2.0.0 py_1**                     |
| **importlib_metadata 2.0.0 1**                        |
| **iniconfig 1.1.1 pyh9f0ad1d_0 conda-forge**          |
| **intel-openmp 2020.2 254**                           |
| **intervaltree 3.1.0 py_0**                           |
| **ipykernel 5.3.4 py38h5ca1d4c_0**                    |
| **ipython 7.20.0 py38hb070fc8_1**                     |
| **ipython_genutils 0.2.0 pyhd3eb1b0_1**               |
| **isort 5.7.0 pyhd3eb1b0_0**                          |
| **jedi 0.17.2 py38h06a4308_1**                        |
| **jeepney 0.6.0 pyhd3eb1b0_0**                        |
| **jinja2 2.11.3 pyhd3eb1b0_0**                        |
| **joblib 1.0.1 pypi_0 pypi**                          |
| **jpeg 9b h024ee3a_2**                                |
| **json5 0.9.5 pyh9f0ad1d_0 conda-forge**              |
| **jsonschema 3.2.0 py_2**                             |
| **jupyter-packaging 0.7.12 pyhd8ed1ab_0 conda-forge** |
| **jupyter_client 6.1.7 py_0**                         |
| **jupyter_core 4.7.1 py38h06a4308_0**                 |
| **jupyter_server 1.4.1 py38h578d9bd_0 conda-forge**   |
| **jupyterlab 3.0.9 pyhd8ed1ab_0 conda-forge**         |
| **jupyterlab_pygments 0.1.2 py_0**                    |
| **jupyterlab_server 2.3.0 pyhd8ed1ab_0 conda-forge**  |
| **keyring 22.0.1 py38h06a4308_0**                     |
| **kiwisolver 1.3.1 py38h2531618_0**                   |
| **lame 3.100 h14c3975_1001 conda-forge**              |
| **lazy-object-proxy 1.5.2 py38h27cfd23_0**            |
| **lcms2 2.11 h396b838_0**                             |
| **ld_impl_linux-64 2.33.1 h53a641e_7**                |
| **libedit 3.1.20191231 h14c3975_1**                   |
| **libffi 3.3 he6710b0_2**                             |
| **libgcc-ng 9.1.0 hdf63c60_0**                        |
| **libgfortran-ng 7.5.0 h14aa051_18 conda-forge**      |
| **libgfortran4 7.5.0 h14aa051_18 conda-forge**        |
| **libiconv 1.16 h516909a_0 conda-forge**              |
| **libpng 1.6.37 hbc83047_0**                          |
| **libsodium 1.0.18 h7b6447c_0**                       |
| **libspatialindex 1.9.3 he6710b0_0**                  |
| **libstdcxx-ng 9.1.0 hdf63c60_0**                     |
| **libtiff 4.1.0 h2733197_1**                          |
| **libuuid 1.0.3 h1bed415_2**                          |
| **libuv 1.40.0 h7b6447c_0**                           |
| **libxcb 1.14 h7b6447c_0**                            |
| **libxml2 2.9.10 hb55368b_3**                         |
| **lz4-c 1.9.3 h2531618_0**                            |
| **markupsafe 1.1.1 py38h7b6447c_0**                   |
| **matplotlib 3.3.4 py38h06a4308_0**                   |
| **matplotlib-base 3.3.4 py38h62a2d02_0**              |
| **mccabe 0.6.1 py38_1**                               |
| **mistune 0.8.4 py38h7b6447c_1000**                   |
| **mkl 2020.2 256**                                    |
| **mkl-service 2.3.0 py38he904b0f_0**                  |
| **mkl_fft 1.2.1 py38h54f3939_0**                      |
| **mkl_random 1.1.1 py38h0573a6f_0**                   |
| **more-itertools 8.7.0 pyhd8ed1ab_0 conda-forge**     |
| **mypy_extensions 0.4.3 py38_0**                      |
| **nbclassic 0.2.6 pyhd8ed1ab_0 conda-forge**          |
| **nbclient 0.5.2 pyhd3eb1b0_0**                       |
| **nbconvert 6.0.7 py38_0**                            |
| **nbformat 5.1.2 pyhd3eb1b0_1**                       |
| **ncurses 6.2 he6710b0_1**                            |
| **nest-asyncio 1.5.1 pyhd3eb1b0_0**                   |
| **nettle 3.6 he412f7d_0 conda-forge**                 |
| **ninja 1.10.2 py38hff7bd54_0**                       |
| **notebook 6.2.0 py38h578d9bd_0 conda-forge**         |
| **numpy 1.19.2 py38h54aff64_0**                       |
| **numpy-base 1.19.2 py38hfa32c7d_0**                  |
| **numpydoc 1.1.0 pyhd3eb1b0_1**                       |
| **olefile 0.46 py_0**                                 |
| **openh264 2.1.1 h8b12597_0 conda-forge**             |
| **openssl 1.1.1k h27cfd23_0**                         |
| **packaging 20.9 pyhd3eb1b0_0**                       |
| **pandas 1.2.2 py38ha9443f7_0**                       |
| **pandoc 2.11 hb0f4dca_0**                            |
| **pandocfilters 1.4.3 py38h06a4308_1**                |
| **parso 0.7.0 py_0**                                  |
| **pathspec 0.7.0 py_0**                               |
| **pcre 8.44 he6710b0_0**                              |
| **pexpect 4.8.0 pyhd3eb1b0_3**                        |
| **pickleshare 0.7.5 pyhd3eb1b0_1003**                 |
| **pillow 8.1.0 py38he98fc37_0**                       |
| **pip 21.0.1 py38h06a4308_0**                         |
| **pluggy 0.13.1 py38_0**                              |
| **prometheus_client 0.9.0 pyhd3deb0d_0 conda-forge**  |
| **prompt-toolkit 3.0.8 py_0**                         |
| **psutil 5.8.0 py38h27cfd23_1**                       |
| **ptyprocess 0.7.0 pyhd3eb1b0_2**                     |
| **py 1.10.0 pyhd3deb0d_0 conda-forge**                |
| **pycodestyle 2.6.0 pyhd3eb1b0_0**                    |
| **pycparser 2.20 py_2**                               |
| **pydocstyle 5.1.1 py_0**                             |
| **pyflakes 2.2.0 pyhd3eb1b0_0**                       |
| **pyglet 1.5.15 py38h578d9bd_0 conda-forge**          |
| **pygments 2.7.4 pyhd3eb1b0_0**                       |
| **pylint 2.6.0 py38_0**                               |
| **pyls-black 0.4.6 hd3eb1b0_0**                       |
| **pyls-spyder 0.3.0 pyhd3eb1b0_0**                    |
| **pyopenssl 20.0.1 pyhd3eb1b0_1**                     |
| **pyparsing 2.4.7 pyhd3eb1b0_0**                      |
| **pyqt 5.9.2 py38h05f1152_4**                         |
| **pyrsistent 0.17.3 py38h7b6447c_0**                  |
| **pysocks 1.7.1 py38h06a4308_0**                      |
| **pytest 6.2.3 py38h578d9bd_0 conda-forge**           |
| **pytest-arraydiff 0.3 py_0 conda-forge**             |
| **pytest-astropy 0.8.0 pyhd3eb1b0_0**                 |
| **pytest-astropy-header 0.1.2 py_0 conda-forge**      |
| **pytest-doctestplus 0.9.0 pyhd8ed1ab_0 conda-forge** |
| **pytest-openfiles 0.5.0 py_0 conda-forge**           |
| **pytest-remotedata 0.3.2 pyh9f0ad1d_0 conda-forge**  |
| **python 3.8.5 h7579374_1**                           |
| **python-dateutil 2.8.1 pyhd3eb1b0_0**                |
| **python-jsonrpc-server 0.4.0 py_0**                  |
| **python-language-server 0.36.2 pyhd3eb1b0_0**        |
| **python_abi 3.8 1_cp38 conda-forge**                 |
| **pytorch 1.8.1 py3.8_cuda10.2_cudnn7.6.5_0 pytorch** |
| **pytz 2021.1 pyhd3eb1b0_0**                          |
| **pyxdg 0.27 pyhd3eb1b0_0**                           |
| **pyyaml 5.4.1 py38h27cfd23_1**                       |
| **pyzmq 20.0.0 py38h2531618_1**                       |
| **qdarkstyle 2.8.1 py_0**                             |
| **qt 5.9.7 h5867ecd_1**                               |
| **qtawesome 1.0.1 py_0**                              |
| **qtconsole 5.0.2 pyhd3eb1b0_0**                      |
| **qtpy 1.9.0 py_0**                                   |
| **readline 8.1 h27cfd23_0**                           |
| **regex 2020.11.13 py38h27cfd23_0**                   |
| **requests 2.25.1 pyhd3eb1b0_0**                      |
| **rope 0.18.0 py_0**                                  |
| **rtree 0.9.4 py38_1**                                |
| **scipy 1.6.0 py38h91f5cce_0**                        |
| **secretstorage 3.3.0 py38h06a4308_0**                |
| **send2trash 1.5.0 py_0 conda-forge**                 |
| **setuptools 52.0.0 py38h06a4308_0**                  |
| **sip 4.19.13 py38he6710b0_0**                        |
| **six 1.15.0 py38h06a4308_0**                         |
| **sniffio 1.2.0 py38h578d9bd_1 conda-forge**          |
| **snowballstemmer 2.1.0 pyhd3eb1b0_0**                |
| **sortedcontainers 2.3.0 pyhd3eb1b0_0**               |
| **soupsieve 2.0.1 py_1 conda-forge**                  |
| **sphinx 3.4.3 pyhd3eb1b0_0**                         |
| **sphinxcontrib-applehelp 1.0.2 pyhd3eb1b0_0**        |
| **sphinxcontrib-devhelp 1.0.2 pyhd3eb1b0_0**          |
| **sphinxcontrib-htmlhelp 1.0.3 pyhd3eb1b0_0**         |
| **sphinxcontrib-jsmath 1.0.1 pyhd3eb1b0_0**           |
| **sphinxcontrib-qthelp 1.0.3 pyhd3eb1b0_0**           |
| **sphinxcontrib-serializinghtml 1.1.4 pyhd3eb1b0_0**  |
| **spyder 4.2.1 py38h06a4308_1**                       |
| **spyder-kernels 1.10.1 py38h06a4308_0**              |
| **sqlite 3.33.0 h62c20be_0**                          |
| **terminado 0.9.2 py38h578d9bd_0 conda-forge**        |
| **testpath 0.4.4 pyhd3eb1b0_0**                       |
| **textdistance 4.2.1 pyhd3eb1b0_0**                   |
| **three-merge 0.1.1 pyhd3eb1b0_0**                    |
| **tk 8.6.10 hbc83047_0**                              |
| **toml 0.10.1 py_0**                                  |
| **torchaudio 0.8.1 py38 pytorch**                     |
| **torchvision 0.9.1 py38_cu102 pytorch**              |
| **tornado 6.1 py38h27cfd23_0**                        |
| **traitlets 5.0.5 pyhd3eb1b0_0**                      |
| **typed-ast 1.4.2 py38h27cfd23_1**                    |
| **typing_extensions 3.7.4.3 pyha847dfd_0**            |
| **tzdata 2020f h52ac0ba_0**                           |
| **ujson 4.0.2 py38h2531618_0**                        |
| **urllib3 1.26.3 pyhd3eb1b0_0**                       |
| **watchdog 1.0.2 py38h06a4308_1**                     |
| **wcwidth 0.2.5 py_0**                                |
| **webencodings 0.5.1 py38_1**                         |
| **wheel 0.36.2 pyhd3eb1b0_0**                         |
| **wrapt 1.12.1 py38h7b6447c_1**                       |
| **wurlitzer 2.0.1 py38_0**                            |
| **x264 1!152.20180806 h14c3975_0 conda-forge**        |
| **xz 5.2.5 h7b6447c_0**                               |
| **yaml 0.2.5 h7b6447c_0**                             |
| **yapf 0.30.0 py_0**                                  |
| **zeromq 4.3.3 he6710b0_3**                           |
| **zipp 3.4.0 pyhd3eb1b0_0**                           |
| **zlib 1.2.11 h7b6447c_3**                            |
| **zstd 1.4.5 h9ceee32_0**                             |

**Unzip the attached TrajectoryBot file and use it as your working
directory**

**Navigate to gym-push in your terminal or Python IDE**

**type “pip install -e .” This will install the custom gym environment
for the spacecraft.**

**Restart the kernel**

**Now you can run “mainPPO.py” from the working directory.**
