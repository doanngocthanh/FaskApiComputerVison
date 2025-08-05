import re
from typing import Dict, Any, Optional
from datetime import datetime


class MRZParser:
    """
    Service class for parsing MRZ (Machine Readable Zone) strings into human-readable information.
    Supports Vietnam ID card MRZ format.
    """
    
    def __init__(self):
        """Initialize MRZ parser."""
        pass
    
    def parse_mrz_string(self, mrz_string: str) -> Dict[str, Any]:
        """
        Parse MRZ string into human-readable information.
        
        Args:
            mrz_string: MRZ string to parse
            
        Returns:
            Dictionary containing parsed information
        """
        try:
            # Clean and validate MRZ string
            mrz_clean = re.sub(r'[^A-Z0-9<]', '', mrz_string.upper().strip())
            
            if not mrz_clean:
                return {
                    "status": "error",
                    "message": "Empty MRZ string",
                    "data": {}
                }
            
            # Detect MRZ format and parse accordingly
            if mrz_clean.startswith('ID'):
                return self._parse_vietnam_id_mrz(mrz_clean)
            elif mrz_clean.startswith('P'):
                return self._parse_passport_mrz(mrz_clean)
            else:
                return self._parse_generic_mrz(mrz_clean)
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error parsing MRZ: {str(e)}",
                "data": {}
            }
    
    def _parse_vietnam_id_mrz(self, mrz_string: str) -> Dict[str, Any]:
        """
        Parse Vietnam ID card MRZ format.
        
        Format example: IDVNM2010029147087201002914<<80110130M2610139VNM<<<<<<<<4HA<<TRUONG<GIANG<<<<<<<<<<<<<<
        Line 1: IDVNM + Document Number + Check Digit + Date of Birth + Check Digit
        Line 2: Sex + Expiry Date + Country Code + Personal Number + Check Digit + Name
        """
        try:
            # Split into potential lines if it's a continuous string
            # Try to identify line breaks based on patterns
            if len(mrz_string) > 60:  # Likely two lines combined
                # Look for pattern that indicates second line start (usually after country code)
                # Vietnam ID typically: Line1 ends with <<, Line2 starts with date/sex
                
                # Find potential split points
                line1_pattern = r'(ID[A-Z]{3}\d+<<?\d*)'
                line2_pattern = r'(\d{6}[MF]\d{6}[A-Z]{3}.*)'
                
                match1 = re.search(line1_pattern, mrz_string)
                match2 = re.search(line2_pattern, mrz_string)
                
                if match1 and match2:
                    line1 = match1.group(1)
                    line2 = match2.group(1)
                else:
                    # Fallback: try to split at reasonable position
                    mid_point = len(mrz_string) // 2
                    # Look for good split point around middle
                    for i in range(mid_point - 10, mid_point + 10):
                        if i < len(mrz_string) and re.match(r'\d{6}[MF]', mrz_string[i:i+7]):
                            line1 = mrz_string[:i]
                            line2 = mrz_string[i:]
                            break
                    else:
                        # Default split
                        line1 = mrz_string[:44]
                        line2 = mrz_string[44:]
            else:
                # Already single line or short string
                line1 = mrz_string
                line2 = ""
            
            # Parse Line 1: IDVNM + Document Number + Optional Check + Date of Birth + Optional Check
            parsed_data = {}
            
            # Extract document type and country
            if line1.startswith('ID'):
                parsed_data['document_type'] = 'ID'
                parsed_data['issuing_country'] = line1[2:5]  # VNM
                
                # Extract document number (after country code until first non-digit/letter)
                remainder = line1[5:]
                doc_num_match = re.match(r'(\d+)', remainder)
                if doc_num_match:
                    parsed_data['document_number'] = doc_num_match.group(1)
                    pos_after_doc = len(doc_num_match.group(1))
                    
                    # Look for date of birth pattern in remaining string
                    remaining = remainder[pos_after_doc:]
                    dob_match = re.search(r'(\d{6})', remaining)
                    if dob_match:
                        dob_str = dob_match.group(1)
                        parsed_data['date_of_birth'] = self._parse_date(dob_str, 'YYMMDD')
                        parsed_data['date_of_birth_raw'] = dob_str
            
            # Parse Line 2 if available: Sex + Expiry Date + Country + Personal Number + Name
            if line2:
                # Look for sex and expiry date pattern
                sex_exp_match = re.match(r'(\d{6})([MF])(\d{6})([A-Z]{3})', line2)
                if sex_exp_match:
                    parsed_data['date_of_birth_2'] = self._parse_date(sex_exp_match.group(1), 'DDMMYY')
                    parsed_data['sex'] = 'Male' if sex_exp_match.group(2) == 'M' else 'Female'
                    parsed_data['expiry_date'] = self._parse_date(sex_exp_match.group(3), 'DDMMYY')
                    parsed_data['nationality'] = sex_exp_match.group(4)
                    
                    # Extract name from remaining part
                    name_part = line2[len(sex_exp_match.group(0)):]
                    parsed_data['full_name'] = self._parse_name(name_part)
                
                # Alternative parsing if above pattern doesn't match
                elif len(line2) > 15:
                    # Look for sex indicator
                    sex_match = re.search(r'([MF])', line2)
                    if sex_match:
                        parsed_data['sex'] = 'Male' if sex_match.group(1) == 'M' else 'Female'
                    
                    # Look for name patterns
                    name_match = re.search(r'[A-Z]{2,}<<[A-Z<]+', line2)
                    if name_match:
                        parsed_data['full_name'] = self._parse_name(name_match.group(0))
            
            # Look for name in the entire string if not found
            if 'full_name' not in parsed_data:
                name_match = re.search(r'[A-Z]{2,}<<[A-Z<]+', mrz_string)
                if name_match:
                    parsed_data['full_name'] = self._parse_name(name_match.group(0))
            
            return {
                "status": "success",
                "message": "MRZ parsed successfully",
                "mrz_type": "Vietnam ID Card",
                "raw_mrz": mrz_string,
                "parsed_lines": {
                    "line1": line1,
                    "line2": line2
                },
                "data": parsed_data
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error parsing Vietnam ID MRZ: {str(e)}",
                "data": {}
            }
    
    def _parse_passport_mrz(self, mrz_string: str) -> Dict[str, Any]:
        """Parse passport MRZ format (placeholder for future implementation)."""
        return {
            "status": "info",
            "message": "Passport MRZ parsing not implemented yet",
            "mrz_type": "Passport",
            "raw_mrz": mrz_string,
            "data": {}
        }
    
    def _parse_generic_mrz(self, mrz_string: str) -> Dict[str, Any]:
        """Parse generic MRZ format."""
        try:
            parsed_data = {}
            
            # Extract basic patterns
            # Look for country codes
            country_match = re.search(r'([A-Z]{3})', mrz_string)
            if country_match:
                parsed_data['country_code'] = country_match.group(1)
            
            # Look for dates (6 digit patterns)
            date_matches = re.findall(r'(\d{6})', mrz_string)
            if date_matches:
                parsed_data['dates_found'] = [self._parse_date(d, 'YYMMDD') for d in date_matches]
                parsed_data['dates_raw'] = date_matches
            
            # Look for sex
            sex_match = re.search(r'([MF])', mrz_string)
            if sex_match:
                parsed_data['sex'] = 'Male' if sex_match.group(1) == 'M' else 'Female'
            
            # Look for names
            name_match = re.search(r'[A-Z]{2,}<<[A-Z<]+', mrz_string)
            if name_match:
                parsed_data['full_name'] = self._parse_name(name_match.group(0))
            
            return {
                "status": "success",
                "message": "Generic MRZ parsing completed",
                "mrz_type": "Generic",
                "raw_mrz": mrz_string,
                "data": parsed_data
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error parsing generic MRZ: {str(e)}",
                "data": {}
            }
    
    def _parse_date(self, date_str: str, format_type: str) -> Optional[str]:
        """
        Parse date string into readable format.
        
        Args:
            date_str: Date string (6 digits)
            format_type: 'YYMMDD' or 'DDMMYY'
            
        Returns:
            Formatted date string or None
        """
        try:
            if len(date_str) != 6:
                return None
            
            if format_type == 'YYMMDD':
                year = int(date_str[:2])
                month = int(date_str[2:4])
                day = int(date_str[4:6])
                
                # Handle 2-digit year (assume 1900s for 50-99, 2000s for 00-49)
                if year >= 50:
                    year += 1900
                else:
                    year += 2000
                    
            elif format_type == 'DDMMYY':
                day = int(date_str[:2])
                month = int(date_str[2:4])
                year = int(date_str[4:6])
                
                # Handle 2-digit year
                if year >= 50:
                    year += 1900
                else:
                    year += 2000
            else:
                return None
            
            # Validate date
            if 1 <= month <= 12 and 1 <= day <= 31:
                return f"{day:02d}/{month:02d}/{year}"
            else:
                return f"{date_str} (invalid date)"
                
        except:
            return None
    
    def _parse_name(self, name_str: str) -> Dict[str, str]:
        """
        Parse name string from MRZ format.
        
        Args:
            name_str: Name string with << separators
            
        Returns:
            Dictionary with parsed name components
        """
        try:
            # Remove extra < characters and split by <<
            name_clean = re.sub(r'<+', '<', name_str.strip('<'))
            name_parts = [part for part in name_clean.split('<') if part]
            
            if not name_parts:
                return {"full_name": name_str}
            
            # First part is usually surname, rest are given names
            if len(name_parts) >= 2:
                surname = name_parts[0]
                given_names = ' '.join(name_parts[1:])
                full_name = f"{given_names} {surname}"
                
                return {
                    "full_name": full_name,
                    "surname": surname,
                    "given_names": given_names,
                    "name_parts": name_parts
                }
            else:
                return {
                    "full_name": name_parts[0],
                    "name_parts": name_parts
                }
                
        except Exception as e:
            return {"full_name": name_str, "error": str(e)}
    
    def validate_mrz_checksum(self, data: str, check_digit: str) -> bool:
        """
        Validate MRZ checksum (basic implementation).
        
        Args:
            data: Data to validate
            check_digit: Expected check digit
            
        Returns:
            Boolean indicating if checksum is valid
        """
        try:
            # MRZ checksum calculation weights: 7, 3, 1 repeating
            weights = [7, 3, 1]
            total = 0
            
            for i, char in enumerate(data):
                if char.isdigit():
                    value = int(char)
                elif char.isalpha():
                    value = ord(char) - ord('A') + 10
                else:  # < character
                    value = 0
                
                total += value * weights[i % 3]
            
            calculated_check = total % 10
            return str(calculated_check) == check_digit
            
        except:
            return False
