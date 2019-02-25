#!groovy

pipeline {
    // Configure the default agent
    agent any
    environment {
        AGAVE_JOB_TIMEOUT = 900
        AGAVE_JOB_GET_DIR = "job_output"
        AGAVE_DATA_URI    = "agave://data-sd2e-community/"
        AGAVE_CACHE_DIR   = "${HOME}/credentials_cache/${JOB_BASE_NAME}"
        AGAVE_JSON_PARSER = "jq"
        AGAVE_TENANTID    = "sd2e"
        AGAVE_APISERVER   = "https://api.sd2e.org"
        AGAVE_USERNAME    = "sd2etest"
        AGAVE_PASSWORD    = credentials('sd2etest-tacc-password')
        REGISTRY_USERNAME = "sd2etest"
        REGISTRY_PASSWORD = credentials('sd2etest-dockerhub-password')
        REGISTRY_ORG      = credentials('sd2etest-dockerhub-org')
        GITLAB_API_AUTH_TOKEN = credentials('sd2etest-gitlab-token')

        PATH = "${HOME}/bin:${HOME}/sd2e-cloud-cli/bin:${env.PATH}"

    }
    stages {

        stage('Run perovskite test harness') {
            when {
                environment name:'gitlabActionType', value:'PUSH'
            }
            steps {
                sh 'echo $GITLAB_API_AUTH_TOKEN'
                sh 'echo "running perovskite test harnesst"'
                sh 'pip install -r requirements-app.txt --user'
                sh "python ${WORKSPACE}/scripts/perovskite_test_harness.py --gitlab_auth $GITLAB_API_AUTH_TOKEN"
                sh 'echo "finished running test harness script"'
            }
        }
    }
    post {

      failure {
           slackSend (message: ":bomb: *${env.JOB_NAME}/${env.BUILD_NUMBER}* failed \n(<${env.BUILD_URL}|Link>)", channel: "#versioned-datasets")
         }
      success {
          slackSend (message: ":white_check_mark: *${env.JOB_NAME}/${env.BUILD_NUMBER}* completed \n(<${env.BUILD_URL}|Link>)", channel: "#versioned-datasets")
      }

    }
}