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
        PATH = "${HOME}/bin:${HOME}/sd2e-cloud-cli/bin:${env.PATH}"

    }
    stages {
        stage('Run test sbatch') {
            steps {
                sh 'echo "running test harness script"'
                sh 'pip install -r requirements.txt --user'
                sh "sbatch  ${WORKSPACE}/scripts/slurm_runner.slurm"
                sh 'echo "finished running test harness"'
            }
        }
    }
    post {
      failure {
           slackSend (message: ":bomb: *${env.JOB_NAME}/${env.BUILD_NUMBER}* failed \n(<${env.BUILD_URL}|Link>)", channel: "@nleiby")
         }
      success {
          slackSend (message: ":white_check_mark: *${env.JOB_NAME}/${env.BUILD_NUMBER}* completed \n(<${env.BUILD_URL}|Link>)", channel: "@nleiby")
      }

    }
}