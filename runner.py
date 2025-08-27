# runner.py
import os
import logging
from datetime import datetime
from typing import List
from config import GameConfig
from competition import GameCompetition

class ExperimentRunner:
    def __init__(self, gemini_api_key: str = None):
        self.competition = GameCompetition()
        
        if gemini_api_key:
            os.environ['GEMINI_API_KEY'] = gemini_api_key
        elif not os.getenv('GEMINI_API_KEY'):
            raise ValueError("Gemini API key required")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def run_thesis_experiments(self, output_dir: str = "thesis_results", 
                              num_games_per_config: int = 20) -> None:
        """Run complete thesis experiments"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Static games: test number of players (3 vs 5)
        static_games = ['salop', 'spulber']
        static_configs = [
            GameConfig(number_of_players=3, number_of_rounds=1),
            GameConfig(number_of_players=5, number_of_rounds=1)
        ]
        
        # Dynamic games: test time horizon (10 vs 50 rounds)
        dynamic_games = ['green_porter', 'athey_bagwell']
        dynamic_configs = [
            GameConfig(number_of_players=3, number_of_rounds=10),
            GameConfig(number_of_players=3, number_of_rounds=50)
        ]
        
        challenger_models = [
            'gemini-1.5-pro',
            'gemini-1.5-flash'
        ]
        
        # Run static game experiments
        self.logger.info("Running static game experiments...")
        static_results = self.competition.run_tournament(
            static_games, challenger_models, static_configs, 
            num_games_per_config=num_games_per_config
        )
        
        # Run dynamic game experiments
        self.logger.info("Running dynamic game experiments...")
        dynamic_results = self.competition.run_tournament(
            dynamic_games, challenger_models, dynamic_configs,
            num_games_per_config=num_games_per_config
        )
        
        # Combine and export results
        all_results = {
            'experiment_results': {
                **static_results['experiment_results'],
                **dynamic_results['experiment_results']
            },
            'comprehensive_metrics': {
                **static_results['comprehensive_metrics'],
                **dynamic_results['comprehensive_metrics']
            }
        }
        
        self.competition.export_results(all_results, output_dir)
        self.logger.info(f"Thesis experiments completed. Results in {output_dir}/")
    
    def run_quick_test(self, game_name: str = 'salop', num_games: int = 2) -> bool:
        """Run quick test to verify setup"""
        
        try:
            self.logger.info(f"Running quick test: {game_name}")
            
            config = GameConfig(number_of_players=3, number_of_rounds=1)
            
            test_results = self.competition.run_tournament(
                [game_name], 
                ['gemini-1.5-flash'],
                [config],
                num_games_per_config=num_games
            )
            
            if test_results['comprehensive_metrics']:
                self.logger.info("Quick test successful!")
                return True
            else:
                self.logger.error("Quick test failed - no metrics generated")
                return False
                
        except Exception as e:
            self.logger.error(f"Quick test failed: {e}")
            return False

def main():
    """Main execution function"""
    
    runner = ExperimentRunner()
    
    # Run quick test first
    if runner.run_quick_test():
        # Run full thesis experiments
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        runner.run_thesis_experiments(
            output_dir=f"thesis_results_{timestamp}",
            num_games_per_config=20
        )
    else:
        print("Quick test failed. Check API key and configuration.")

if __name__ == "__main__":
    main()