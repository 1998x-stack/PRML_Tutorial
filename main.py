structure = {
    "1_Introduction": [
        {
            "1.1_Example:_Polynomial_Curve_Fitting": []
        },
        {
            "1.2_Probability_Theory": [
                "1.2.1_Probability_densities",
                "1.2.2_Expectations_and_covariances",
                "1.2.3_Bayesian_probabilities",
                "1.2.4_The_Gaussian_distribution",
                "1.2.5_Curve_fitting_re-visited",
                "1.2.6_Bayesian_curve_fitting"
            ]
        },
        "1.3_Model_Selection",
        "1.4_The_Curse_of_Dimensionality",
        {
            "1.5_Decision_Theory": [
                "1.5.1_Minimizing_the_misclassification_rate",
                "1.5.2_Minimizing_the_expected_loss",
                "1.5.3_The_reject_option",
                "1.5.4_Inference_and_decision",
                "1.5.5_Loss_functions_for_regression"
            ]
        },
        {
            "1.6_Information_Theory": [
                "1.6.1_Relative_entropy_and_mutual_information"
            ]
        },
    ],
    "2_Probability_Distributions": [
        {
            "2.1_Binary_Variables": [
                "2.1.1_The_beta_distribution"
            ]
        },
        {
            "2.2_Multinomial_Variables": [
                "2.2.1_The_Dirichlet_distribution"
            ]
        },
        {
            "2.3_The_Gaussian_Distribution": [
                "2.3.1_Conditional_Gaussian_distributions",
                "2.3.2_Marginal_Gaussian_distributions",
                "2.3.3_Bayes’_theorem_for_Gaussian_variables",
                "2.3.4_Maximum_likelihood_for_the_Gaussian",
                "2.3.5_Sequential_estimation",
                "2.3.6_Bayesian_inference_for_the_Gaussian",
                "2.3.7_Student’s_t-distribution",
                "2.3.8_Periodic_variables",
                "2.3.9_Mixtures_of_Gaussians"
            ]
        },
        {
            "2.4_The_Exponential_Family": [
                "2.4.1_Maximum_likelihood_and_sufficient_statistics",
                "2.4.2_Conjugate_priors",
                "2.4.3_Noninformative_priors"
            ]
        },
        {
            "2.5_Nonparametric_Methods": [
                "2.5.1_Kernel_density_estimators",
                "2.5.2_Nearest-neighbour_methods"
            ]
        },
    ],
    "3_Linear_Models_for_Regression": [
        {
            "3.1_Linear_Basis_Function_Models": [
                "3.1.1_Maximum_likelihood_and_least_squares",
                "3.1.2_Geometry_of_least_squares",
                "3.1.3_Sequential_learning",
                "3.1.4_Regularized_least_squares",
                "3.1.5_Multiple_outputs"
            ]
        },
        "3.2_The_Bias-Variance_Decomposition",
        {
            "3.3_Bayesian_Linear_Regression": [
                "3.3.1_Parameter_distribution",
                "3.3.2_Predictive_distribution",
                "3.3.3_Equivalent_kernel"
            ]
        },
        "3.4_Bayesian_Model_Comparison",
        {
            "3.5_The_Evidence_Approximation": [
                "3.5.1_Evaluation_of_the_evidence_function",
                "3.5.2_Maximizing_the_evidence_function",
                "3.5.3_Effective_number_of_parameters"
            ]
        },
        "3.6_Limitations_of_Fixed_Basis_Functions",
    ],
    "4_Linear_Models_for_Classification": [
        {
            "4.1_Discriminant_Functions": [
                "4.1.1_Two_classes",
                "4.1.2_Multiple_classes",
                "4.1.3_Least_squares_for_classification",
                "4.1.4_Fisher’s_linear_discriminant",
                "4.1.5_Relation_to_least_squares",
                "4.1.6_Fisher’s_discriminant_for_multiple_classes",
                "4.1.7_The_perceptron_algorithm"
            ]
        },
        {
            "4.2_Probabilistic_Generative_Models": [
                "4.2.1_Continuous_inputs",
                "4.2.2_Maximum_likelihood_solution",
                "4.2.3_Discrete_features",
                "4.2.4_Exponential_family"
            ]
        },
        {
            "4.3_Probabilistic_Discriminative_Models": [
                "4.3.1_Fixed_basis_functions",
                "4.3.2_Logistic_regression",
                "4.3.3_Iterative_reweighted_least_squares",
                "4.3.4_Multiclass_logistic_regression",
                "4.3.5_Probit_regression",
                "4.3.6_Canonical_link_functions"
            ]
        },
        {
            "4.4_The_Laplace_Approximation": [
                "4.4.1_Model_comparison_and_BIC"
            ]
        },
        {
            "4.5_Bayesian_Logistic_Regression": [
                "4.5.1_Laplace_approximation",
                "4.5.2_Predictive_distribution"
            ]
        },
    ],
    "5_Neural_Networks": [
        {
            "5.1_Feed-forward_Network_Functions": [
                "5.1.1_Weight-space_symmetries"
            ]
        },
        {
            "5.2_Network_Training": [
                "5.2.1_Parameter_optimization",
                "5.2.2_Local_quadratic_approximation",
                "5.2.3_Use_of_gradient_information",
                "5.2.4_Gradient_descent_optimization"
            ]
        },
        {
            "5.3_Error_Backpropagation": [
                "5.3.1_Evaluation_of_error-function_derivatives",
                "5.3.2_A_simple_example",
                "5.3.3_Efficiency_of_backpropagation",
                "5.3.4_The_Jacobian_matrix"
            ]
        },
        {
            "5.4_The_Hessian_Matrix": [
                "5.4.1_Diagonal_approximation",
                "5.4.2_Outer_product_approximation",
                "5.4.3_Inverse_Hessian",
                "5.4.4_Finite_differences",
                "5.4.5_Exact_evaluation_of_the_Hessian",
                "5.4.6_Fast_multiplication_by_the_Hessian"
            ]
        },
        {
            "5.5_Regularization_in_Neural_Networks": [
                "5.5.1_Consistent_Gaussian_priors",
                "5.5.2_Early_stopping",
                "5.5.3_Invariances",
                "5.5.4_Tangent_propagation",
                "5.5.5_Training_with_transformed_data",
                "5.5.6_Convolutional_networks",
                "5.5.7_Soft_weight_sharing"
            ]
        },
        "5.6_Mixture_Density_Networks",
        {
            "5.7_Bayesian_Neural_Networks": [
                "5.7.1_Posterior_parameter_distribution",
                "5.7.2_Hyperparameter_optimization",
                "5.7.3_Bayesian_neural_networks_for_classification"
            ]
        },
    ],
    "6_Kernel_Methods": [
        "6.1_Dual_Representations",
        "6.2_Constructing_Kernels",
        {
            "6.3_Radial_Basis_Function_Networks": [
                "6.3.1_Nadaraya-Watson_model"
            ]
        },
        {
            "6.4_Gaussian_Processes": [
                "6.4.1_Linear_regression_revisited",
                "6.4.2_Gaussian_processes_for_regression",
                "6.4.3_Learning_the_hyperparameters",
                "6.4.4_Automatic_relevance_determination",
                "6.4.5_Gaussian_processes_for_classification",
                "6.4.6_Laplace_approximation",
                "6.4.7_Connection_to_neural_networks"
            ]
        },
    ],
    "7_Sparse_Kernel_Machines": [
        {
            "7.1_Maximum_Margin_Classifiers": [
                "7.1.1_Overlapping_class_distributions",
                "7.1.2_Relation_to_logistic_regression",
                "7.1.3_Multiclass_SVMs",
                "7.1.4_SVMs_for_regression",
                "7.1.5_Computational_learning_theory"
            ]
        },
        {
            "7.2_Relevance_Vector_Machines": [
                "7.2.1_RVM_for_regression",
                "7.2.2_Analysis_of_sparsity",
                "7.2.3_RVM_for_classification"
            ]
        },
    ],
    "8_Graphical_Models": [
        {
            "8.1_Bayesian_Networks": [
                "8.1.1_Example:_Polynomial_regression",
                "8.1.2_Generative_models",
                "8.1.3_Discrete_variables",
                "8.1.4_Linear-Gaussian_models"
            ]
        },
        {
            "8.2_Conditional_Independence": [
                "8.2.1_Three_example_graphs",
                "8.2.2_D-separation"
            ]
        },
        {
            "8.3_Markov_Random_Fields": [
                "8.3.1_Conditional_independence_properties",
                "8.3.2_Factorization_properties",
                "8.3.3_Illustration:_Image_de-noising",
                "8.3.4_Relation_to_directed_graphs"
            ]
        },
        {
            "8.4_Inference_in_Graphical_Models": [
                "8.4.1_Inference_on_a_chain",
                "8.4.2_Trees",
                "8.4.3_Factor_graphs",
                "8.4.4_The_sum-product_algorithm",
                "8.4.5_The_max-sum_algorithm",
                "8.4.6_Exact_inference_in_general_graphs",
                "8.4.7_Loopy_belief_propagation",
                "8.4.8_Learning_the_graph_structure"
            ]
        },
    ],
    "9_Mixture_Models_and_EM": [
        {
            "9.1_K-means_Clustering": [
                "9.1.1_Image_segmentation_and_compression"
            ]
        },
        {
            "9.2_Mixtures_of_Gaussians": [
                "9.2.1_Maximum_likelihood",
                "9.2.2_EM_for_Gaussian_mixtures"
            ]
        },
        {
            "9.3_An_Alternative_View_of_EM": [
                "9.3.1_Gaussian_mixtures_revisited",
                "9.3.2_Relation_to_K-means",
                "9.3.3_Mixtures_of_Bernoulli_distributions",
                "9.3.4_EM_for_Bayesian_linear_regression"
            ]
        },
        "9.4_The_EM_Algorithm_in_General",
    ],
    "10_Approximate_Inference": [
        {
            "10.1_Variational_Inference": [
                "10.1.1_Factorized_distributions",
                "10.1.2_Properties_of_factorized_approximations",
                "10.1.3_Example:_The_univariate_Gaussian",
                "10.1.4_Model_comparison"
            ]
        },
        {
            "10.2_Illustration:_Variational_Mixture_of_Gaussians": [
                "10.2.1_Variational_distribution",
                "10.2.2_Variational_lower_bound",
                "10.2.3_Predictive_density",
                "10.2.4_Determining_the_number_of_components",
                "10.2.5_Induced_factorizations"
            ]
        },
        {
            "10.3_Variational_Linear_Regression": [
                "10.3.1_Variational_distribution",
                "10.3.2_Predictive_distribution",
                "10.3.3_Lower_bound"
            ]
        },
        {
            "10.4_Exponential_Family_Distributions": [
                "10.4.1_Variational_message_passing"
            ]
        },
        "10.5_Local_Variational_Methods",
        {
            "10.6_Variational_Logistic_Regression": [
                "10.6.1_Variational_posterior_distribution",
                "10.6.2_Optimizing_the_variational_parameters",
                "10.6.3_Inference_of_hyperparameters"
            ]
        },
        {
            "10.7_Expectation_Propagation": [
                "10.7.1_Example:_The_clutter_problem",
                "10.7.2_Expectation_propagation_on_graphs"
            ]
        },
    ],
    "11_Sampling_Methods": [
        {
            "11.1_Basic_Sampling_Algorithms": [
                "11.1.1_Standard_distributions",
                "11.1.2_Rejection_sampling",
                "11.1.3_Adaptive_rejection_sampling",
                "11.1.4_Importance_sampling",
                "11.1.5_Sampling-importance-resampling",
                "11.1.6_Sampling_and_the_EM_algorithm"
            ]
        },
        {
            "11.2_Markov_Chain_Monte_Carlo": [
                "11.2.1_Markov_chains",
                "11.2.2_The_Metropolis-Hastings_algorithm"
            ]
        },
        "11.3_Gibbs_Sampling",
        "11.4_Slice_Sampling",
        {
            "11.5_The_Hybrid_Monte_Carlo_Algorithm": [
                "11.5.1_Dynamical_systems",
                "11.5.2_Hybrid_Monte_Carlo"
            ]
        },
        "11.6_Estimating_the_Partition_Function",
    ],
    "12_Continuous_Latent_Variables": [
        {
            "12.1_Principal_Component_Analysis": [
                "12.1.1_Maximum_variance_formulation",
                "12.1.2_Minimum-error_formulation",
                "12.1.3_Applications_of_PCA",
                "12.1.4_PCA_for_high-dimensional_data"
            ]
        },
        {
            "12.2_Probabilistic_PCA": [
                "12.2.1_Maximum_likelihood_PCA",
                "12.2.2_EM_algorithm_for_PCA",
                "12.2.3_Bayesian_PCA",
                "12.2.4_Factor_analysis"
            ]
        },
        "12.3_Kernel_PCA",
        {
            "12.4_Nonlinear_Latent_Variable_Models": [
                "12.4.1_Independent_component_analysis",
                "12.4.2_Autoassociative_neural_networks",
                "12.4.3_Modelling_nonlinear_manifolds"
            ]
        },
    ],
    "13_Sequential_Data": [
        "13.1_Markov_Models",
        {
            "13.2_Hidden_Markov_Models": [
                "13.2.1_Maximum_likelihood_for_the_HMM",
                "13.2.2_The_forward-backward_algorithm",
                "13.2.3_The_sum-product_algorithm_for_the_HMM",
                "13.2.4_Scaling_factors",
                "13.2.5_The_Viterbi_algorithm",
                "13.2.6_Extensions_of_the_hidden_Markov_model"
            ]
        },
        {
            "13.3_Linear_Dynamical_Systems": [
                "13.3.1_Inference_in_LDS",
                "13.3.2_Learning_in_LDS",
                "13.3.3_Extensions_of_LDS",
                "13.3.4_Particle_filters"
            ]
        },
    ],
    "14_Combining_Models": [
        "14.1_Bayesian_Model_Averaging",
        "14.2_Committees",
        {
            "14.3_Boosting": [
                "14.3.1_Minimizing_exponential_error",
                "14.3.2_Error_functions_for_boosting"
            ]
        },
        "14.4_Tree-based_Models",
        {
            "14.5_Conditional_Mixture_Models": [
                "14.5.1_Mixtures_of_linear_regression_models",
                "14.5.2_Mixtures_of_logistic_models",
                "14.5.3_Mixtures_of_experts"
            ]
        },
    ]
}

import os
from typing import Dict, Any

def create_directories_and_files(
        base_path: str, 
        structure: Dict[str, Any], 
        readme_file, 
        parent_path: str = "", 
        level: int = 1
    ):
    heading = "#" * level

    for key, value in structure.items():
        current_path = os.path.join(base_path, key.replace(" ", "_").replace("/", "_").replace("-", "_"))

        # 创建目录
        os.makedirs(current_path, exist_ok=True)

        # 在README中添加章节标题
        if parent_path:
            readme_file.write(f"{heading} {parent_path}/{key}\n\n")
        else:
            readme_file.write(f"{heading} {key}\n\n")

        # 递归调用创建子目录和文件
        if isinstance(value, dict) and value:
            create_directories_and_files(
                current_path, 
                value, 
                readme_file, 
                parent_path + "/" + key if parent_path else key, 
                level + 1
            )
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, dict) and item:
                    create_directories_and_files(
                        current_path, 
                        item, 
                        readme_file, 
                        parent_path + "/" + key if parent_path else key, 
                        level + 1
                    )
                else:
                    item = f"{idx:02d}_{item}"
                    file_name = item.replace(" ", "_").replace("/", "_").replace("-", "_") + ".py"
                    file_path = os.path.join(current_path, file_name)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(f"# {item}\n\n")
                        file.write(f'"""\nLecture: {parent_path}/{key}\nContent: {item}\n"""\n\n')

                    # 在README中添加文件链接
                    item_clean = item.replace(" ", "_").replace("/", "_").replace("-", "_")
                    parent_clean = parent_path.replace(" ", "_").replace("/", "_").replace("-", "_")
                    key_clean = key.replace(" ", "_").replace("/", "_").replace("-", "_")
                    readme_file.write(f"- [{item}](./{parent_clean}/{key_clean}/{item_clean}.py)\n")
                    
                    
                    file_name = item.replace(" ", "_").replace("/", "_").replace("-", "_") + ".md"
                    file_path = os.path.join(current_path, file_name)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(f"# {item}\n\n")
                        file.write(f'"""\nLecture: {parent_path}/{key}\nContent: {item}\n"""\n\n')

                    # 在README中添加文件链接
                    item_clean = item.replace(" ", "_").replace("/", "_").replace("-", "_")
                    parent_clean = parent_path.replace(" ", "_").replace("/", "_").replace("-", "_")
                    key_clean = key.replace(" ", "_").replace("/", "_").replace("-", "_")
                    readme_file.write(f"- [{item}](./{parent_clean}/{key_clean}/{item_clean}.md)\n")
        else:
            # 创建文件并写入初始内容
            file_name = key.replace(" ", "_").replace("/", "_").replace("-", "_") + ".py"
            file_path = os.path.join(current_path, file_name)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(f"# {key}\n\n")
                file.write(f'"""\nLecture: {parent_path}/{key}\nContent: {key}\n"""\n\n')

            # 在README中添加文件链接
            parent_clean = parent_path.replace(" ", "_").replace("/", "_").replace("-", "_")
            key_clean = key.replace(" ", "_").replace("/", "_").replace("-", "_")
            readme_file.write(f"- [{key}](./{parent_clean}/{key_clean}/{file_name})\n")
            
            
            file_name = key.replace(" ", "_").replace("/", "_").replace("-", "_") + ".md"
            file_path = os.path.join(current_path, file_name)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(f"# {key}\n\n")
                file.write(f'"""\nLecture: {parent_path}/{key}\nContent: {key}\n"""\n\n')

            # 在README中添加文件链接
            parent_clean = parent_path.replace(" ", "_").replace("/", "_").replace("-", "_")
            key_clean = key.replace(" ", "_").replace("/", "_").replace("-", "_")
            readme_file.write(f"- [{key}](./{parent_clean}/{key_clean}/{file_name})\n")

        # 添加空行以分隔不同的章节
        readme_file.write("\n")

def main():
    root_dir = './'
    # 创建根目录
    os.makedirs(root_dir, exist_ok=True)

    # 创建 README.md 文件
    with open(os.path.join(root_dir, "README.md"), 'w', encoding='utf-8') as readme_file:
        readme_file.write("# PRML\n\n")
        readme_file.write("这是一个关于PRML的目录结构。\n\n")
        create_directories_and_files(root_dir, structure, readme_file)

    print("目录和文件结构已生成，并创建 README.md 文件。")

if __name__ == "__main__":
    main()