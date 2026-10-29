pipeline {
    agent any

    options {
        timeout(time: 45, unit: 'MINUTES')
    }

    stages {
        stage('Validate') {
            steps {
                sh '''
                    set -eu
                    python -m compileall -q backend ingestion experiments
                    cd frontend
                    npm ci
                    npm run build
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
}
