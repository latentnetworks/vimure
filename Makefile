up:
	docker-compose build notebooks
	docker-compose up notebooks

test:
	docker-compose build
	docker-compose run --rm notebooks pytest -s --pyargs vimure #/test/test_model.py::TestVimureModel::test_vimure_model_extreme_scenarios_over_reporting_dense

