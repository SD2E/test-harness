README Last Updated by Hamed on 2/8/19


# Test Harness
###### Current Version: 3.2

## Why use the Test Harness?
1. Consistent results and outputs, both in terms of format and in terms of ensuring results are valid and without bugs
2. Collaboration and awareness of peer analytics through leaderboard summaries and access to model code
3. Automated calculation of various model performance metrics
4. Automated selection of future experiments to run (coming soon for protein design and perovskites)

## Using the Test Harness
You can use the Test Harness locally or with TACC resources such as Maverick.

Python 3 should be used when using the Test Harness.

### Installation
1. Clone this repository into the environment of your choice
2. Using command-line, navigate to the directory in which you cloned this repo.
3. Run `pip install -e test-harness` , this will install the `test-harness` package and make
it visible to all other repositories/projects you have in the current environment.
(The `-e` makes it possible for pip to install a local package not available on pypi.)


### Running The Test Harness
1. Create a script in your environment.
2. Import modules from the `test-harness` package you installed earlier. For example,
`from harness.test_harness_class import TestHarness`
3. Results will be output in a `test_harness_results` folder whose location depends on the path you give to `output_location` when initializing your TestHarness object.


### Executing Runs:
In your script file (see `example_script.py` for an example), the steps are as follows:
1. Create a TestHarness object with output path
2. Use the `run_custom` or `run_leave_one_out` methods
3. View results in the `results` folder

### Run Options
There are 2 types of runs currently supported in the Test Harness: Custom Runs and Leave-One-Out Runs
More details coming soon (probably in pydoc format)

#### Feature Extraction Options
Feature Extraction is controlled with the `feature_extraction` option. Valid values are:
1. False --> Feature Extraction is not carried out
2. True --> The `eli5_permutation` method is used
3. "eli5_permutation" --> Uses the [eli5 package's implementation of permutation importance for SkLearn Models](https://eli5.readthedocs.io/en/latest/autodocs/sklearn.html#eli5.sklearn.permutation_importance.PermutationImportance).
A more general version that isn't specific to sklearn models will be added soon.
4. "rfpimp_permutation" --> Uses the [rfpimp package's implementation of permutation importance for Random Forest Models only](https://github.com/parrt/random-forest-importances).

Feature Extraction results are saved in the file `feature_importances.csv` in the appropriate subfolder within the `results` folder.
Note: In a test that I ran, I got significantly different results from the two different implementations of permutation importance: this needs to be investigated.


### Maverick Instructions:

#### Setup:
1. `ssh _____@maverick.tacc.utexas.edu`
2. Enter user/pass/auth-code
3. `module load tacc-singularity/2.6.0`
    1. Not necessary if you make singularity part of your default set of modules using "module save"
4. Only do these ONCE: first get on a compute node by using the `idev` command. Then run `singularity pull docker://eram5/test-harness:3.2`
    1. This is only needed the first time you’re pulling the image, or if you want to pull an updated version of the image.
    2. This will build a singularity image from the docker container and store it in a ".simg" file under singularity_cache
    3. The path to the ".simg" file will be printed out, make sure to save the path for everytime you need to kick off a singularity shell!


#### Running interactively with `idev`:
1. `cd [wherever you're working]`. e.g. `cd $WORK` (`cdw` is equivalent) followed by `cd protein-design`
2. `idev -m 100` [or however many minutes you want the session to last]
    1. Note: if you want to keep a session going after you exit terminal (or get disconnected), you could use the `screen` command before `idev`
3. `singularity shell --nv /work/05260/hamed/singularity_cache/test-harness-3.2.simg` --> path is different for each user!
    1. The path is the one that was printed when you first pulled the docker container
    2. This starts an interactive singularity shell and places you within it
4. `python3 setup.py install --user`
    1. You will need to rerun this anytime there are major changes to code or file structure. Usually if you change code within a script file you won't need to rerun this.
5. `python3 [name_of_my_script_file]` after navigating to the `scripts` folder/subfolders. e.g. `python3 example_script.py` from within `test_harness/scripts/examples`.

Note: if you try to install things within the singularity container, you probably will have to add the “--user” parameter at the end
e.g. pip install pandas --user
Ideally you would install all requirements within your dockerfile though

#### Running non-interactively using Sbatch/Slurm:
View the `sbatch_example.slurm` file in the `test_harness/scripts/examples` folder. As you can see the `sbatch_example.slurm` file is pretty self-explanatory.
In the last line you can see the command that runs `example_script.py`. In this case there is just one command being run, but you can add more if you would like.
Just keep in mind that the commands will be run in serial on a single node. To access multiple nodes and parallelization, you can either manually kick off more
jobs in the same way, or you can use launcher (see below for the guide to that).

1. The command to run a `slurm` file is `sbatch sbatch_example.slurm`. 
2. You can view the progress of your job in realtime by using `tail -f [name_of_my_job_id_file]`

#### Running non-interactively using Sbatch/Slurm, combined with parallelization using Launcher and Argparse:
Looking at the test_harness/scripts/examples` folder, there are 3 files that are used for this:
1. `example_script_argparse.py` --> the script I am running
2. `jobfile_example` --> just a list of the jobs I want to run. Make sure you have an empty line at the end of this file or your last line won't be read!
3. `launcher_example.slurm` --> slurm file that uses Launcher to organize and split up tasks in the `jobfile` over multiple nodes so they can run in parallel

Once again you would use the `sbatch` and `tail` commands to kick off your jobs and view their progress.

**Please note that if you have major changes to the codebase, you will have to reinstall setup.py. The current way to do this is to get on a
compute node interactively using `idev`, start a singularity shell, and run `python3 setup.py install --user`. Then you can exit the shell and compute node,
and resume using sbatch/slurm/launcher. I am working with Joe Allen to remove this minor inconvenience soon.

### How It Works (Behind the Scenes)
The Test Harness consists of several parts that work together. 
1. Test Harness **Model** Classes and Instances: Basically just wrappers for your ML model so that the Test Harness can know how to run your model.
    1. All TH-Model Classes must subclass from the abstract base classes in `test_harness_models_abstract_classes.py`. A Model Class defines how to fit and predict.
    2. Once a TH-Model Class is created (e.g. SklearnClassification), TH-Model Instances can be created with specific models passed in.
2. `test_harness_class.py` contains the `TestHarness` class, which allows the user to add and execute different types of runs (custom, leave-one-out).
3. `run_classes.py` defines classes for different types of runs. These are used by the `TestHarness` class to carry out its operations.
4.  Script files that run the Test Harness. This is where you implement the Test Harness and use its methods to run your models.

As a user, you will mainly be concerned with the Test Harness `model classes/instances` (bullet 1 above), and the script files (bullet 4 above).


## FAQ
Q: Can we see other people's results and models?
A: The TACC App for the Test Harness is currently down. We will be working to get it up soon and that will allow for this.





