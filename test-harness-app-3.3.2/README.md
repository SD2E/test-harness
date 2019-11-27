# Deploying Updating Version

- Change the app version in CONTAINER_IMAGE
- Change the app version in app.ini
- Change the version in version.py
- Change the version name in the folder test-harness-app-X.X.X
- Clear out the test_harness_results folder


Using the appropriate version number, run 

`docker build -f Dockerfile -t test-harness-app:3.3.2 ./`
#`docker push nleiby/test-harness-app:3.3.1`


- `docker build -f Dockerfile --force-rm -t nleiby-test-harness:3.3.2 ./`
- Or, if your local env is clean, `docker build -f Dockerfile -t nleiby-test-harness:3.3.2 ./`

- `apps-deploy test-harness-app-3.3.2` # This also pushes the updated image to dockerhub


# Updating dependencies

The test harness is invoked in the versioned data repo at `scripts/run_test_harness_on_perovskites.py`.  Need to update the `TEST_HARNESS_REACTOR_NAME` there.