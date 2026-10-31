pipeline {
    agent any

    triggers {
        githubPush()
    }

    environment {
        SUCCESS_EMAIL_RECIPIENTS = 'suryasatishganiga@gmail.com'
    }

    options {
        timeout(time: 45, unit: 'MINUTES')
    }

    stages {
        stage('Validate') {
            steps {
                sh '''
                    set -eu
                    if command -v python >/dev/null 2>&1; then
                        PYTHON_CMD=python
                    elif command -v python3 >/dev/null 2>&1; then
                        PYTHON_CMD=python3
                    else
                        apt-get update
                        apt-get install -y python3
                        PYTHON_CMD=python3
                    fi
                    "$PYTHON_CMD" -m compileall -q backend ingestion experiments
                    cd frontend
                    if ! command -v npm >/dev/null 2>&1; then
                        apt-get update
                        apt-get install -y nodejs npm
                    fi
                    npm ci
                    npm run build
                    npx playwright install chromium
                    npm run test:e2e
                '''
            }
        }

        stage('Build and publish images') {
            when {
                branch 'main'
            }
            steps {
                withCredentials([
                    usernamePassword(
                        credentialsId: 'dockerhub-credentials',
                        usernameVariable: 'DOCKERHUB_USERNAME',
                        passwordVariable: 'DOCKERHUB_TOKEN'
                    )
                ]) {
                    sh '''
                        set -eu
                        IMAGE_TAG="sha-$(git rev-parse HEAD)"
                        trap 'docker logout >/dev/null 2>&1 || true' EXIT
                        printf '%s' "$DOCKERHUB_TOKEN" | docker login \
                            --username "$DOCKERHUB_USERNAME" --password-stdin

                        docker build --pull -f Dockerfile.backend \
                            -t "$DOCKERHUB_USERNAME/mlops-repe-backend:$IMAGE_TAG" \
                            -t "$DOCKERHUB_USERNAME/mlops-repe-backend:latest" .
                        docker build --pull -f Dockerfile.collector \
                            -t "$DOCKERHUB_USERNAME/mlops-repe-collector:$IMAGE_TAG" \
                            -t "$DOCKERHUB_USERNAME/mlops-repe-collector:latest" .
                        docker build --pull --build-arg VITE_API_BASE= \
                            -f frontend/Dockerfile \
                            -t "$DOCKERHUB_USERNAME/mlops-repe-frontend:$IMAGE_TAG" \
                            -t "$DOCKERHUB_USERNAME/mlops-repe-frontend:latest" frontend

                        docker push "$DOCKERHUB_USERNAME/mlops-repe-backend:$IMAGE_TAG"
                        docker push "$DOCKERHUB_USERNAME/mlops-repe-backend:latest"
                        docker push "$DOCKERHUB_USERNAME/mlops-repe-collector:$IMAGE_TAG"
                        docker push "$DOCKERHUB_USERNAME/mlops-repe-collector:latest"
                        docker push "$DOCKERHUB_USERNAME/mlops-repe-frontend:$IMAGE_TAG"
                        docker push "$DOCKERHUB_USERNAME/mlops-repe-frontend:latest"
                    '''
                }
            }

        }

        stage('Deploy to Kubernetes') {
            when {
                branch 'main'
            }
            steps {
                withCredentials([
                    usernamePassword(
                        credentialsId: 'dockerhub-credentials',
                        usernameVariable: 'DOCKERHUB_USERNAME',
                        passwordVariable: 'DOCKERHUB_TOKEN'
                    ),
                    file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')
                ]) {
                    sh '''
                        set -eu
                        IMAGE_TAG="sha-$(git rev-parse HEAD)"
                        rendered_manifest="$(mktemp)"
                        trap 'rm -f "$rendered_manifest"' EXIT

                        sed \
                            -e "s|DOCKERHUB_USERNAME|$DOCKERHUB_USERNAME|g" \
                            -e "s|IMAGE_TAG|$IMAGE_TAG|g" \
                            k8s/application.yaml > "$rendered_manifest"

                        kubectl apply -f "$rendered_manifest"
                        kubectl rollout status deployment/mlops-repe-backend \
                            --namespace mlops-repe --timeout=180s
                        kubectl rollout status deployment/mlops-repe-frontend \
                            --namespace mlops-repe --timeout=180s
                    '''
                }

            }
        }
    }

    post {
        success {
            script {
                def successRecipients = env.SUCCESS_EMAIL_RECIPIENTS?.trim() ?: 'suryasatishganiga@gmail.com'
                if (successRecipients) {
                    emailext(
                        subject: "SUCCESS: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                        body: """<h2>Build successful</h2>
<p><b>Project:</b> ${env.JOB_NAME}</p>
<p><b>Build:</b> #${env.BUILD_NUMBER}</p>
<p><b>Branch:</b> ${env.GIT_BRANCH ?: 'unknown'}</p>
<p><b>Commit:</b> ${env.GIT_COMMIT ?: 'unknown'}</p>
<p><b>Console:</b> <a href="${env.BUILD_URL}">${env.BUILD_URL}</a></p>
<p>Python validation, frontend build, and 12 Playwright website tests passed.</p>""",
                        to: successRecipients,
                        mimeType: 'text/html'
                    )
                } else {
                    echo 'Successful-build recipients are empty; email skipped.'
                }
            }
        }
    }
}
