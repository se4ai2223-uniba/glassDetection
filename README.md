Glasses Detection
==============================

[![quality_assurance_workflow](https://github.com/se4ai2223-uniba/glassDetection/actions/workflows/QA.yml/badge.svg)](https://github.com/se4ai2223-uniba/glassDetection/actions/workflows/QA.yml)

[![api_testing_workflow](https://github.com/se4ai2223-uniba/glassDetection/actions/workflows/api-test.yaml/badge.svg)](https://github.com/se4ai2223-uniba/glassDetection/actions/workflows/api-test.yaml)

[![test-code](https://github.com/se4ai2223-uniba/glassDetection/actions/workflows/test_code.yml/badge.svg)](https://github.com/se4ai2223-uniba/glassDetection/actions/workflows/test_code.yml)

[![dvc repro](https://github.com/se4ai2223-uniba/glassDetection/actions/workflows/dvc_repro.yml/badge.svg)](https://github.com/se4ai2223-uniba/glassDetection/actions/workflows/dvc_repro.yml)

[![CI/CD](https://github.com/se4ai2223-uniba/glassDetection/actions/workflows/CI_CD.yml/badge.svg)](https://github.com/se4ai2223-uniba/glassDetection/actions/workflows/CI_CD.yml)

[![server_status](https://github.com/se4ai2223-uniba/glassDetection/actions/workflows/server_check.yml/badge.svg)](https://yfvpqbuhav.eu-west-1.awsapprunner.com/)



Deploy of a ML (CNN - based) model able to detect subjects wearing glass in photos.

Project Organization
------------
    📦glassDetection
     ┣ 📂.dvc
     ┣ 📂.git
     ┣ 📂.github
     ┃ ┗ 📂workflows
     ┃ ┃ ┣ 📜api-test.yaml
     ┃ ┃ ┣ 📜CI_CD.yml
     ┃ ┃ ┣ 📜dvc_repro.yml
     ┃ ┃ ┣ 📜QA.yml
     ┃ ┃ ┣ 📜server_check.yml
     ┃ ┃ ┗ 📜test_code.yml
     ┣ 📂app
     ┃ ┣ 📜api.py
     ┃ ┣ 📜Dockerfile
     ┃ ┣ 📜monitoring.py
     ┃ ┣ 📜schemas.py
     ┃ ┣ 📜test2.jpg
     ┃ ┣ 📜test_api.py
     ┃ ┣ 📜test_img.jpeg
     ┃ ┣ 📜test_img.jpg
     ┃ ┗ 📜test_server.py
     ┣ 📂app_frontend
     ┃ ┣ 📂test
     ┃ ┣ 📜Dockerfile
     ┃ ┣ 📜interface.py
     ┃ ┗ 📜requirements_frontend.txt
     ┣ 📂data
     ┃ ┣ 📂CelebA
     ┃ ┃ ┣ 📂figures
     ┃ ┃ ┃ ┗ 📜exampleCelebA.png
     ┃ ┃ ┣ 📂raw
     ┃ ┃ ┃ ┣ 📜.gitignore
     ┃ ┃ ┃ ┣ 📜celeb_augmented_preprocessed.h5.dvc
     ┃ ┃ ┃ ┣ 📜list_attr_celeba.txt
     ┃ ┃ ┃ ┗ 📜noglass_celeb_augmented_preprocessed.h5.dvc
     ┃ ┃ ┗ 📜README.md
     ┃ ┣ 📂Selfie
     ┃ ┃ ┣ 📂figures
     ┃ ┃ ┃ ┣ 📜sample.jpg
     ┃ ┃ ┃ ┗ 📜selfie_dataset6.jpg
     ┃ ┃ ┣ 📂processed
     ┃ ┃ ┃ ┗ 📜selfie_dataset.txt
     ┃ ┃ ┣ 📂raw
     ┃ ┃ ┃ ┣ 📜.gitignore
     ┃ ┃ ┃ ┣ 📜noglass_selfie_augmented_preprocessed.h5.dvc
     ┃ ┃ ┃ ┗ 📜selfie_augmented_preprocessed.h5.dvc
     ┃ ┃ ┗ 📜README.md
     ┃ ┣ 📂Selfie_reduced
     ┃ ┃ ┣ 📂processed
     ┃ ┃ ┃ ┣ 📂images
     ┃ ┃ ┃ ┣ 📜.gitignore
     ┃ ┃ ┃ ┣ 📜README.txt
     ┃ ┃ ┃ ┣ 📜selfie_reduced.h5
     ┃ ┃ ┃ ┗ 📜selfie_reduced.h5.dvc
     ┃ ┃ ┣ 📂raw
     ┃ ┃ ┃ ┣ 📜.gitignore
     ┃ ┃ ┃ ┣ 📜Selfie-dataset.zip
     ┃ ┃ ┃ ┣ 📜Selfie-dataset.zip.dvc
     ┃ ┃ ┃ ┗ 📜selfie_reduced.h5
     ┃ ┃ ┗ 📜.gitignore
     ┃ ┗ 📂UTK_face
     ┃ ┃ ┣ 📂figures
     ┃ ┃ ┃ ┗ 📜samples.png
     ┃ ┃ ┣ 📂raw
     ┃ ┃ ┃ ┣ 📜.gitignore
     ┃ ┃ ┃ ┣ 📜noglass_utk_augmented_preprocessed.h5.dvc
     ┃ ┃ ┃ ┗ 📜utk_augmented_preprocessed.h5.dvc
     ┃ ┃ ┗ 📜README.md
     ┣ 📂docs
     ┃ ┣ 📜commands.rst
     ┃ ┣ 📜conf.py
     ┃ ┣ 📜getting-started.rst
     ┃ ┣ 📜index.rst
     ┃ ┣ 📜make.bat
     ┃ ┗ 📜Makefile
     ┣ 📂great_expectations
     ┃ ┣ 📂checkpoints
     ┃ ┃ ┣ 📜13_rows_deleted.yml
     ┃ ┃ ┗ 📜13_rows_deleted1.yml
     ┃ ┣ 📂expectations
     ┃ ┃ ┣ 📜.ge_store_backend_id
     ┃ ┃ ┣ 📜reduced.json
     ┃ ┃ ┗ 📜reduced_suite.json
     ┃ ┣ 📂plugins
     ┃ ┃ ┗ 📂custom_data_docs
     ┃ ┃ ┃ ┣ 📂renderers
     ┃ ┃ ┃ ┣ 📂styles
     ┃ ┃ ┃ ┃ ┗ 📜data_docs_custom_styles.css
     ┃ ┃ ┃ ┗ 📂views
     ┃ ┣ 📜.gitignore
     ┃ ┗ 📜great_expectations.yml
     ┣ 📂models
     ┃ ┣ 📂figures
     ┃ ┃ ┣ 📜hardToClassify.png
     ┃ ┃ ┣ 📜ourCNN.png
     ┃ ┃ ┣ 📜pipeline.PNG
     ┃ ┃ ┗ 📜stateOfTheArtPerformance.png
     ┃ ┣ 📜finalModelGlassDetection255.dvc
     ┃ ┗ 📜README.md
     ┣ 📂monitoring
     ┃ ┣ 📜drift-detection.py
     ┃ ┣ 📜locustfile.py
     ┃ ┗ 📜prometheus.yml
     ┣ 📂notebooks
     ┃ ┗ 📜.gitkeep
     ┣ 📂references
     ┃ ┗ 📜.gitkeep
     ┣ 📂reports
     ┃ ┣ 📂figures
     ┃ ┃ ┗ 📜.gitkeep
     ┃ ┗ 📜.gitkeep
     ┣ 📂src
     ┃ ┣ 📂data
     ┃ ┃ ┣ 📜.gitkeep
     ┃ ┃ ┣ 📜FaceAlignerNetwork.py
     ┃ ┃ ┣ 📜make_dataset.py
     ┃ ┃ ┗ 📜__init__.py
     ┃ ┣ 📂features
     ┃ ┃ ┣ 📜.gitkeep
     ┃ ┃ ┣ 📜build_features.py
     ┃ ┃ ┗ 📜__init__.py
     ┃ ┣ 📂models
     ┃ ┃ ┣ 📜.gitkeep
     ┃ ┃ ┣ 📜predict_model.py
     ┃ ┃ ┣ 📜shape_predictor_5_face_landmarks.dat
     ┃ ┃ ┣ 📜train_model.py
     ┃ ┃ ┗ 📜__init__.py
     ┃ ┣ 📂visualization
     ┃ ┃ ┣ 📜.gitkeep
     ┃ ┃ ┣ 📜visualize.py
     ┃ ┃ ┗ 📜__init__.py
     ┃ ┗ 📜__init__.py
     ┣ 📂tests
     ┃ ┣ 📜.gitignore
     ┃ ┣ 📜testing_data.py
     ┃ ┣ 📜test_make_dataset.py
     ┃ ┗ 📜test_model.py
     ┣ 📜.dvcignore
     ┣ 📜.gitignore
     ┣ 📜docker-compose.yml
     ┣ 📜drift-detection.py
     ┣ 📜dvc.lock
     ┣ 📜dvc.yaml
     ┣ 📜LICENSE
     ┣ 📜Makefile
     ┣ 📜README.md
     ┣ 📜requirements.txt
     ┣ 📜setup.py
     ┣ 📜test_environment.py
     ┗ 📜tox.ini
     
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
