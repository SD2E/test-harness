# Deploying Updating Version

- Change the app version in CONTAINER_IMAGE
- Change the app version in app.ini
- Change the version in version.py
- Change the version name in the folder test-harness-app-X.X.X
- Clear out the test_harness_results folder


Using the appropriate version number, run 

`docker build -f Dockerfile -t test-harness-app:X.X.X ./`
#`docker push nleiby/test-harness-app:X.X.X`


- `docker build -f Dockerfile --force-rm -t nleiby-test-harness:X.X.X ./`
- Or, if your local env is clean, `docker build -f Dockerfile -t nleiby-test-harness:X.X.X ./`

- `apps-deploy test-harness-app-X.X.X` # This also pushes the updated image to dockerhub


# Updating dependencies

The test harness is invoked in the versioned data repo at `scripts/run_test_harness_on_perovskites.py`.  Need to update the `TEST_HARNESS_REACTOR_NAME` there.