"""
LOINC Mapper Module

This module provides utilities for mapping laboratory tests to LOINC codes.
LOINC (Logical Observation Identifiers Names and Codes) is a universal standard
for identifying medical laboratory observations.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class LOINCMapper:
    """Maps laboratory tests to LOINC codes and retrieves related information."""
    
    def __init__(self, loinc_file_path: str = None):
        """Initialize LOINC mapper with mapping data.
        
        Args:
            loinc_file_path: Path to LOINC mapping JSON file
        """
        if loinc_file_path is None:
            loinc_file_path = Path(__file__).parent.parent / "data" / "loinc_mapping.json"
        
        with open(loinc_file_path, 'r') as f:
            data = json.load(f)
            self.mappings = data['loinc_mappings']
            self.metadata = data.get('metadata', {})
    
    def get_loinc_code(self, test_name: str) -> Optional[str]:
        """Get LOINC code for a test name.
        
        Args:
            test_name: Name of the laboratory test
            
        Returns:
            LOINC code or None if not found
        """
        if test_name in self.mappings:
            return self.mappings[test_name]['loinc_code']
        return None
    
    def get_test_info(self, test_name: str) -> Optional[Dict]:
        """Get complete information for a laboratory test.
        
        Args:
            test_name: Name of the laboratory test
            
        Returns:
            Dictionary with test information or None if not found
        """
        return self.mappings.get(test_name)
    
    def search_by_loinc_code(self, loinc_code: str) -> Optional[Dict]:
        """Search for test information by LOINC code.
        
        Args:
            loinc_code: LOINC code to search for
            
        Returns:
            Dictionary with test information or None if not found
        """
        for test_name, info in self.mappings.items():
            if info['loinc_code'] == loinc_code:
                return {'test_name': test_name, **info}
        return None
    
    def get_synonyms(self, test_name: str) -> List[str]:
        """Get synonyms for a laboratory test.
        
        Args:
            test_name: Name of the laboratory test
            
        Returns:
            List of synonyms
        """
        if test_name in self.mappings:
            return self.mappings[test_name].get('synonyms', [])
        return []
    
    def search_by_keyword(self, keyword: str) -> List[Dict]:
        """Search for tests by keyword in names and synonyms.
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            List of matching tests with their information
        """
        keyword_lower = keyword.lower()
        results = []
        
        for test_name, info in self.mappings.items():
            # Check test name
            if keyword_lower in test_name.lower():
                results.append({'test_name': test_name, **info})
                continue
            
            # Check long common name
            if keyword_lower in info.get('long_common_name', '').lower():
                results.append({'test_name': test_name, **info})
                continue
            
            # Check synonyms
            for synonym in info.get('synonyms', []):
                if keyword_lower in synonym.lower():
                    results.append({'test_name': test_name, **info})
                    break
        
        return results
    
    def get_all_tests(self) -> List[str]:
        """Get list of all test names in the mapping.
        
        Returns:
            List of test names
        """
        return list(self.mappings.keys())
    
    def get_all_loinc_codes(self) -> List[str]:
        """Get list of all LOINC codes in the mapping.
        
        Returns:
            List of LOINC codes
        """
        return [info['loinc_code'] for info in self.mappings.values()]


if __name__ == '__main__':
    # Example usage
    mapper = LOINCMapper()
    
    print("LOINC Mapper Demo")
    print("=" * 50)
    
    # Get LOINC code for glucose
    test = 'glucose_in_blood'
    code = mapper.get_loinc_code(test)
    print(f"\nLOINC code for '{test}': {code}")
    
    # Get full information
    info = mapper.get_test_info(test)
    print(f"\nFull information for '{test}':")
    print(f"  Long name: {info['long_common_name']}")
    print(f"  Component: {info['component']}")
    print(f"  System: {info['system']}")
    print(f"  Synonyms: {', '.join(info['synonyms'])}")
    
    # Search by keyword
    print("\n\nSearching for 'blood':")
    results = mapper.search_by_keyword('blood')
    for result in results[:3]:
        print(f"  - {result['test_name']}: {result['loinc_code']}")
    
    # Search by LOINC code
    print("\n\nSearching for LOINC code '1975-2':")
    result = mapper.search_by_loinc_code('1975-2')
    if result:
        print(f"  Test: {result['test_name']}")
        print(f"  Name: {result['long_common_name']}")
