import numpy as np
from urllib.parse import urlparse, parse_qs
from pyalex import Works, Authors, Institutions
import pandas as pd
import ast, json
from typing import Optional, Callable

def openalex_url_to_pyalex_query(url):
    """
    Convert an OpenAlex search URL to a pyalex query.
    
    Args:
    url (str): The OpenAlex search URL.
    
    Returns:
    tuple: (Works object, dict of parameters)
    """
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    # Initialize the Works object
    query = Works()
    
    # Handle filters
    if 'filter' in query_params:
        filters = query_params['filter'][0].split(',')
        for f in filters:
            if ':' in f:
                key, value = f.split(':', 1)
                if key == 'default.search':
                    query = query.search(value)
                else:
                    query = query.filter(**{key: value})
    
    # Handle sort - Fixed to properly handle field:direction format
    if 'sort' in query_params:
        sort_params = query_params['sort'][0].split(',')
        for s in sort_params:
            if ':' in s:  # Handle field:direction format
                field, direction = s.split(':')
                query = query.sort(**{field: direction})
            elif s.startswith('-'):  # Handle -field format
                query = query.sort(**{s[1:]: 'desc'})
            else:  # Handle field format
                query = query.sort(**{s: 'asc'})
    
    # Handle other parameters
    params = {}
    for key in ['page', 'per-page', 'sample', 'seed']:
        if key in query_params:
            params[key] = query_params[key][0]
    
    return query, params

def invert_abstract(inv_index):
    """Reconstruct abstract from OpenAlex' inverted-index.

    Handles dicts, JSON / repr strings, or missing values gracefully.
    """
    # Try to coerce a string into a Python object first
    if isinstance(inv_index, str):
        try:
            inv_index = json.loads(inv_index)          # double-quoted JSON
        except Exception:
            try:
                inv_index = ast.literal_eval(inv_index)  # single-quoted repr
            except Exception:
                inv_index = None

    if isinstance(inv_index, dict):
        l_inv = [(w, p) for w, pos in inv_index.items() for p in pos]
        return " ".join(w for w, _ in sorted(l_inv, key=lambda x: x[1]))
    else:
        return " "  
    
    
def get_pub(x):
    """Extract publication name from record."""
    try: 
        source = x['source']['display_name']
        if source not in ['parsed_publication','Deleted Journal']:
            return source
        else: 
            return ' '
    except:
            return ' '

def get_field(x):
    """Extract academic field from record."""
    try:
        field = x['primary_topic']['subfield']['display_name']
        if field is not None:
            return field
        else:
            return np.nan
    except:
        return np.nan

def process_records_to_df(records):
    """
    Convert OpenAlex records to a pandas DataFrame with processed fields.
    Can handle either raw OpenAlex records or an existing DataFrame.
    
    Args:
    records (list or pd.DataFrame): List of OpenAlex record dictionaries or existing DataFrame
    
    Returns:
    pandas.DataFrame: Processed DataFrame with abstracts, publications, and titles
    """
    # If records is already a DataFrame, use it directly
    if isinstance(records, pd.DataFrame):
        records_df = records.copy()
        # Only process abstract_inverted_index and primary_location if they exist
        if 'abstract_inverted_index' in records_df.columns:
            records_df['abstract'] = [invert_abstract(t) for t in records_df['abstract_inverted_index']]
        if 'primary_location' in records_df.columns:
            records_df['parsed_publication'] = [get_pub(x) for x in records_df['primary_location']]
            records_df['parsed_publication'] = records_df['parsed_publication'].fillna(' ') # fill missing values with space, only if we have them.
        
    else:
        # Process raw records as before
        records_df = pd.DataFrame(records)
        records_df['abstract'] = [invert_abstract(t) for t in records_df['abstract_inverted_index']]
        records_df['parsed_publication'] = [get_pub(x) for x in records_df['primary_location']]
        records_df['parsed_publication'] = records_df['parsed_publication'].fillna(' ')
    
    # Fill missing values and deduplicate
    
    records_df['abstract'] = records_df['abstract'].fillna(' ')
    records_df['title'] = records_df['title'].fillna(' ')
    records_df = records_df.drop_duplicates(subset=['id']).reset_index(drop=True)
    
    return records_df

def openalex_url_to_filename(url):
    """
    Convert an OpenAlex URL to a filename-safe string with timestamp.
    
    Args:
    url (str): The OpenAlex search URL
    
    Returns:
    str: A filename-safe string with timestamp (without extension)
    """
    from datetime import datetime
    import re
    
    # First parse the URL into query and params
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    # Create parts of the filename
    parts = []
    
    # Handle filters
    if 'filter' in query_params:
        filters = query_params['filter'][0].split(',')
        for f in filters:
            if ':' in f:
                key, value = f.split(':', 1)
                # Replace dots with underscores and clean the value
                key = key.replace('.', '_')
                # Clean the value to be filename-safe and add spaces around words
                clean_value = re.sub(r'[^\w\s-]', '', value)
                # Replace multiple spaces with single space and strip
                clean_value = ' '.join(clean_value.split())
                # Replace spaces with underscores for filename
                clean_value = clean_value.replace(' ', '_')
                
                if key == 'default_search':
                    parts.append(f"search_{clean_value}")
                else:
                    parts.append(f"{key}_{clean_value}")
    
    # Handle sort parameters
    if 'sort' in query_params:
        sort_params = query_params['sort'][0].split(',')
        for s in sort_params:
            if s.startswith('-'):
                parts.append(f"sort_{s[1:].replace('.', '_')}_desc")
            else:
                parts.append(f"sort_{s.replace('.', '_')}_asc")
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Combine all parts
    filename = '__'.join(parts) if parts else 'openalex_query'
    filename = f"{filename}__{timestamp}"
    
    # Ensure filename is not too long (max 255 chars is common filesystem limit)
    if len(filename) > 255:
        filename = filename[:251]  # leave room for potential extension
    
    return filename

def get_records_from_dois(doi_list, block_size=50):
    """
    Download OpenAlex records for a list of DOIs in blocks.
    Args:
        doi_list (list): List of DOIs (strings)
        block_size (int): Number of DOIs to fetch per request (default 50)
    Returns:
        pd.DataFrame: DataFrame of OpenAlex records
    """
    from pyalex import Works
    from tqdm import tqdm
    all_records = []
    for i in tqdm(range(0, len(doi_list), block_size)):
        sublist = doi_list[i:i+block_size]
        doi_str = "|".join(sublist)
        try:
            record_list = Works().filter(doi=doi_str).get(per_page=block_size)
            all_records.extend(record_list)
        except Exception as e:
            print(f"Error fetching DOIs {sublist}: {e}")
    return pd.DataFrame(all_records)

def openalex_url_to_readable_name(url):
    """
    Convert an OpenAlex URL to a short, human-readable query description.
    
    Args:
    url (str): The OpenAlex search URL
    
    Returns:
    str: A short, human-readable description of the query
    
    Examples:
    - "Search: 'Kuramoto Model'"
    - "Search: 'quantum physics', 2020-2023"
    - "Cites: Popper (1959)"
    - "From: University of Pittsburgh, 1999-2020"
    - "By: Einstein, A., 1905-1955"
    """
    import re
    
    # Parse the URL
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    # Initialize description parts
    parts = []
    year_range = None
    
    # Handle filters
    if 'filter' in query_params:
        filters = query_params['filter'][0].split(',')
        
        for f in filters:
            if ':' not in f:
                continue
                
            key, value = f.split(':', 1)
            
            try:
                if key == 'default.search':
                    # Clean up search term (remove quotes if present)
                    search_term = value.strip('"\'')
                    parts.append(f"Search: '{search_term}'")
                    
                elif key == 'title_and_abstract.search':
                    # Handle title and abstract search specifically
                    from urllib.parse import unquote_plus
                    search_term = unquote_plus(value).strip('"\'')
                    parts.append(f"T&A: '{search_term}'")
                    
                elif key == 'publication_year':
                    # Handle year ranges or single years
                    if '-' in value:
                        start_year, end_year = value.split('-')
                        year_range = f"{start_year}-{end_year}"
                    else:
                        year_range = value
                        
                elif key == 'cites':
                    # Look up the cited work to get author and year
                    work_id = value
                    try:
                        cited_work = Works()[work_id]
                        if cited_work:
                            # Get first author's last name
                            author_name = "Unknown"
                            year = "Unknown"
                            
                            if cited_work.get('authorships') and len(cited_work['authorships']) > 0:
                                first_author = cited_work['authorships'][0]['author']
                                if first_author.get('display_name'):
                                    # Extract last name (assuming "First Last" format)
                                    name_parts = first_author['display_name'].split()
                                    author_name = name_parts[-1] if name_parts else first_author['display_name']
                            
                            if cited_work.get('publication_year'):
                                year = str(cited_work['publication_year'])
                                
                            parts.append(f"Cites: {author_name} ({year})")
                        else:
                            parts.append(f"Cites: Work {work_id}")
                    except Exception as e:
                        print(f"Could not fetch cited work {work_id}: {e}")
                        parts.append(f"Cites: Work {work_id}")
                        
                elif key == 'authorships.institutions.lineage':
                    # Look up institution name
                    inst_id = value
                    try:
                        institution = Institutions()[inst_id]
                        if institution and institution.get('display_name'):
                            parts.append(f"From: {institution['display_name']}")
                        else:
                            parts.append(f"From: Institution {inst_id}")
                    except Exception as e:
                        print(f"Could not fetch institution {inst_id}: {e}")
                        parts.append(f"From: Institution {inst_id}")
                        
                elif key == 'authorships.author.id':
                    # Look up author name
                    author_id = value
                    try:
                        author = Authors()[author_id]
                        if author and author.get('display_name'):
                            parts.append(f"By: {author['display_name']}")
                        else:
                            parts.append(f"By: Author {author_id}")
                    except Exception as e:
                        print(f"Could not fetch author {author_id}: {e}")
                        parts.append(f"By: Author {author_id}")
                        
                elif key == 'type':
                    # Handle work types
                    type_mapping = {
                        'article': 'Articles',
                        'book': 'Books',
                        'book-chapter': 'Book Chapters',
                        'dissertation': 'Dissertations',
                        'preprint': 'Preprints'
                    }
                    work_type = type_mapping.get(value, value.replace('-', ' ').title())
                    parts.append(f"Type: {work_type}")
                    
                elif key == 'host_venue.id':
                    # Look up venue name
                    venue_id = value
                    try:
                        # For venues, we can use Works to get source info, but let's try a direct approach
                        # This might need adjustment based on pyalex API structure
                        parts.append(f"In: Venue {venue_id}")  # Fallback
                    except Exception as e:
                        parts.append(f"In: Venue {venue_id}")
                        
                elif key.startswith('concepts.id'):
                    # Handle concept filters - these are topic/concept IDs
                    concept_id = value
                    parts.append(f"Topic: {concept_id}")  # Could be enhanced with concept lookup
                    
                else:
                    # Generic handling for other filters
                    from urllib.parse import unquote_plus
                    clean_key = key.replace('_', ' ').replace('.', ' ').title()
                    # Properly decode URL-encoded values
                    try:
                        clean_value = unquote_plus(value).replace('_', ' ')
                    except:
                        clean_value = value.replace('_', ' ')
                    parts.append(f"{clean_key}: {clean_value}")
                    
            except Exception as e:
                print(f"Error processing filter {f}: {e}")
                continue
    
    # Combine parts into final description
    if not parts:
        description = "OpenAlex Query"
    else:
        description = ", ".join(parts)
    
    # Add year range if present
    if year_range:
        if parts:
            description += f", {year_range}"
        else:
            description = f"Works from {year_range}"
    
    # Limit length to keep it readable
    if len(description) > 60:
        description = description[:57] + "..."
        
    return description

def download_openalex_records(
    text_input: str,
    reduce_sample: bool = False,
    sample_reduction_method: str = "All",
    sample_size: int = 0,
    seed_value: str = "42",
    progress: Optional[Callable[[float, str], None]] = None,
) -> pd.DataFrame:
    """
    Download OpenAlex records for one or more OpenAlex search URLs separated by ';'.

    Parameters
    ----------
    text_input : str
        One or more OpenAlex URLs separated by ';'.
    reduce_sample : bool, optional
        Whether to reduce the sample size, by default False.
    sample_reduction_method : str, optional
        Sampling method: "All", "First n samples", or "n random samples".
    sample_size : int, optional
        Target sample size when sampling is enabled, by default 0.
    seed_value : str, optional
        Seed value for random sampling, by default "42".
    progress : Optional[Callable[[float, str], None]], optional
        Optional callback for reporting progress. Signature: (value: float, desc: str) -> None.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame of records (via process_records_to_df).

    Raises
    ------
    ValueError
        If the provided text_input is empty or whitespace.
    """
    import time
    import random

    print(f"Input: {text_input}")
    if not text_input or text_input.isspace():
        error_message = (
            "Error: Please enter a valid OpenAlex URL in the 'OpenAlex-search URL'-field or upload a CSV file"
        )
        raise ValueError(error_message)

    # noop progress if none provided
    def _noop_progress(_: float, desc: str = "") -> None:
        return None

    progress_cb = progress or _noop_progress

    print('Starting data projection pipeline')
    progress_cb(0.1, desc="Starting...")

    start_time = time.time()

    # Split input into multiple URLs if present
    urls = [url.strip() for url in text_input.split(';')]
    records: list[dict] = []
    query_indices: list[int] = []  # Track which query each record comes from
    total_query_length = 0
    expected_download_count = 0  # Track expected number of records to download for progress
    
    # Use first URL for filename (returned value not used here, but keep for parity/logs)
    first_query, first_params = openalex_url_to_pyalex_query(urls[0])
    filename = openalex_url_to_filename(urls[0])
    print(f"Filename: {filename}")

    # Process each URL
    for i, url in enumerate(urls):
        query, params = openalex_url_to_pyalex_query(url)
        query_length = query.count()
        total_query_length += query_length
        
        # Calculate expected download count for this query
        if reduce_sample and sample_reduction_method == "First n samples":
            expected_for_this_query = min(sample_size, query_length)
        elif reduce_sample and sample_reduction_method == "n random samples":
            expected_for_this_query = min(sample_size, query_length)
        else:  # "All"
            expected_for_this_query = query_length
        
        expected_download_count += expected_for_this_query
        print(f'Requesting {query_length} entries from query {i+1}/{len(urls)} (expecting to download {expected_for_this_query})...')
        
        # Use PyAlex sampling for random samples - much more efficient!
        if reduce_sample and sample_reduction_method == "n random samples":
            # Use PyAlex's built-in sample method for efficient server-side sampling
            target_size = min(sample_size, query_length)
            try:
                seed_int = int(seed_value) if str(seed_value).strip() else 42
            except ValueError:
                seed_int = 42
                print(f"Invalid seed value '{seed_value}', using default: 42")
            
            print(f'Attempting PyAlex sampling: {target_size} from {query_length} (seed={seed_int})')
            
            try:
                # Check if PyAlex sample method exists and works
                if hasattr(query, 'sample'):
                    sampled_records = []
                    seen_ids = set()  # Track IDs to avoid duplicates
                    
                    # If target_size > 10k, do batched sampling
                    if target_size > 10000:
                        batch_size = 9998  # Use 9998 to stay safely under 10k limit
                        remaining = target_size
                        batch_num = 1
                        
                        print(f'Target size {target_size} > 10k, using batched sampling with batch size {batch_size}')
                        
                        while remaining > 0 and len(sampled_records) < target_size:
                            current_batch_size = min(batch_size, remaining)
                            batch_seed = seed_int + batch_num  # Different seed for each batch
                            
                            print(f'Batch {batch_num}: requesting {current_batch_size} samples (seed={batch_seed})')
                            
                            # Sample this batch
                            batch_query = query.sample(current_batch_size, seed=batch_seed)
                            
                            batch_records = []
                            batch_count = 0
                            for page in batch_query.paginate(per_page=200, method='page', n_max=None):
                                for record in page:
                                    # Check for duplicates using OpenAlex ID
                                    record_id = record.get('id', '')
                                    if record_id not in seen_ids:
                                        seen_ids.add(record_id)
                                        batch_records.append(record)
                                        batch_count += 1
                            
                            sampled_records.extend(batch_records)
                            remaining -= len(batch_records)
                            batch_num += 1
                            
                            print(f'Batch {batch_num-1} complete: got {len(batch_records)} unique records ({len(sampled_records)}/{target_size} total)')
                            
                            progress_cb(0.1 + (0.15 * len(sampled_records) / max(target_size, 1)), 
                                        desc=f"Batched sampling from query {i+1}/{len(urls)}... ({len(sampled_records)}/{target_size})")
                            
                            # Safety check to avoid infinite loops
                            if batch_num > 20:  # Max 20 batches (should handle up to ~200k samples)
                                print("Warning: Maximum batch limit reached, stopping sampling")
                                break
                    else:
                        # Single batch sampling for <= 10k
                        sampled_query = query.sample(target_size, seed=seed_int)
                        
                        records_count = 0
                        for page in sampled_query.paginate(per_page=200, method='page', n_max=None):
                            for record in page:
                                sampled_records.append(record)
                                records_count += 1
                                progress_cb(0.1 + (0.15 * records_count / max(target_size, 1)), 
                                            desc=f"Getting sampled data from query {i+1}/{len(urls)}... ({records_count}/{target_size})")
                    
                    print(f'PyAlex sampling successful: got {len(sampled_records)} records (requested {target_size})')
                else:
                    raise AttributeError("sample method not available")
                    
            except Exception as e:
                print(f"PyAlex sampling failed ({e}), using fallback method...")
                
                # Fallback: get all records and sample manually
                all_records = []
                records_count = 0
                
                # Use page pagination for fallback method
                for page in query.paginate(per_page=200, method='page', n_max=None):
                    for record in page:
                        all_records.append(record)
                        records_count += 1
                        progress_cb(0.1 + (0.15 * records_count / max(query_length, 1)), 
                                    desc=f"Downloading for sampling from query {i+1}/{len(urls)}...")
                        
                # Now sample manually
                if len(all_records) > target_size:
                    random.seed(seed_int)
                    sampled_records = random.sample(all_records, target_size)
                else:
                    sampled_records = all_records
                    
                print(f'Fallback sampling: got {len(sampled_records)} from {len(all_records)} total')
            
            # Add the sampled records
            for idx, record in enumerate(sampled_records):
                records.append(record)
                query_indices.append(i)
                # Safe progress calculation
                if expected_download_count > 0:
                    progress_val = 0.1 + (0.2 * len(records) / expected_download_count)
                else:
                    progress_val = 0.1
                progress_cb(progress_val, desc=f"Processing sampled data from query {i+1}/{len(urls)}...")
        else:
            # Keep existing logic for "First n samples" and "All"
            target_size = sample_size if reduce_sample and sample_reduction_method == "First n samples" else query_length
            records_per_query = 0
            
            print(f"Query {i+1}: target_size={target_size}, query_length={query_length}, method={sample_reduction_method}")
            
            should_break_current_query = False
            # For "First n samples", limit the maximum records fetched to avoid over-downloading
            max_records_to_fetch = target_size if reduce_sample and sample_reduction_method == "First n samples" else None
            for page in query.paginate(per_page=200, method='page', n_max=max_records_to_fetch):
                # Add retry mechanism for processing each page
                max_retries = 5
                base_wait_time = 1  # Starting wait time in seconds
                exponent = 1.5  # Exponential factor
                
                for retry_attempt in range(max_retries):
                    try:
                        for record in page:
                            # Safety check: don't process if we've already reached target
                            if reduce_sample and sample_reduction_method == "First n samples" and records_per_query >= target_size:
                                print(f"Reached target size before processing: {records_per_query}/{target_size}, breaking from download")
                                should_break_current_query = True
                                break
                            
                            records.append(record)
                            query_indices.append(i)  # Track which query this record comes from
                            records_per_query += 1
                            # Safe progress calculation
                            if expected_download_count > 0:
                                progress_val = 0.1 + (0.2 * len(records) / expected_download_count)
                            else:
                                progress_val = 0.1
                            progress_cb(progress_val, desc=f"Getting data from query {i+1}/{len(urls)}...")
                            
                            if reduce_sample and sample_reduction_method == "First n samples" and records_per_query >= target_size:
                                print(f"Reached target size: {records_per_query}/{target_size}, breaking from download")
                                should_break_current_query = True
                                break
                        # If we get here without an exception, break the retry loop
                        break
                    except Exception as e:
                        print(f"Error processing page: {e}")
                        if retry_attempt < max_retries - 1:
                            wait_time = base_wait_time * (exponent ** retry_attempt) + random.random()
                            print(f"Retrying in {wait_time:.2f} seconds (attempt {retry_attempt + 1}/{max_retries})...")
                            time.sleep(wait_time)
                        else:
                            print(f"Maximum retries reached. Continuing with next page.")
                        
                    # Break out of retry loop if we've reached target
                    if should_break_current_query:
                        break
            
            if should_break_current_query:
                print(f"Successfully downloaded target size for query {i+1}, moving to next query")
                # Continue to next query instead of breaking the entire query loop
                continue
        # Continue to next query - don't break out of the main query loop
    print(f"Query completed in {time.time() - start_time:.2f} seconds")
    print(f"Total records collected: {len(records)}")
    print(f"Expected to download: {expected_download_count}")
    print(f"Available from all queries: {total_query_length}")
    print(f"Sample method used: {sample_reduction_method}")
    print(f"Reduce sample enabled: {reduce_sample}")
    if sample_reduction_method == "n random samples":
        print(f"Seed value: {seed_value}")

    # Build DataFrame and process using existing helper
    if len(records) == 0:
        return pd.DataFrame()

    df = process_records_to_df(pd.DataFrame(records))
    # Attach query index for traceability (optional extra info)
    try:
        if len(query_indices) == len(df):
            df["query_index"] = query_indices
    except Exception:
        pass

    return df