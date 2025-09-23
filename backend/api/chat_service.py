import os
import json
import logging
import re
import random
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Current system info
SYSTEM_INFO = {
    "date": "2025-04-06",
    "version": "2.1.0",
    "user": "aaravgoel0"
}

def preprocess_case_data(case_data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean up and structure case data for better prompting."""
    processed_data = {}
    
    # Extract basic info
    processed_data["case_id"] = case_data.get("case_id", "Unknown")
    processed_data["video_path"] = case_data.get("video_path", "Unknown")
    
    # Clean video path
    if processed_data["video_path"]:
        processed_data["video_name"] = os.path.basename(str(processed_data["video_path"]))
    else:
        processed_data["video_name"] = "Unknown"
    
    # Format timestamp
    if "timestamp" in case_data:
        try:
            if isinstance(case_data["timestamp"], str):
                dt = datetime.fromisoformat(case_data["timestamp"].replace('Z', '+00:00'))
                processed_data["timestamp"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                processed_data["timestamp"] = str(case_data.get("timestamp", "Unknown"))
        except (ValueError, AttributeError):
            processed_data["timestamp"] = str(case_data.get("timestamp", "Unknown"))
    
    # Process person data
    persons = case_data.get("person_identities", [])
    processed_data["person_count"] = len(persons)
    
    # Process all persons with complete details
    processed_data["persons"] = []
    
    # Create a sorted list of persons by appearance count for consistent ranking
    sorted_persons = []
    for person in persons:
        person_id = person.get("id")
        person_info = {"id": person_id}
        
        metadata = person.get("metadata", {})
        if metadata:
            # Clean up appearances data
            if "appearances" in metadata:
                person_info["appearances"] = metadata["appearances"]
            else:
                person_info["appearances"] = 0
                
            # Clean up time data
            if "first_seen_time" in metadata and "last_seen_time" in metadata:
                person_info["first_seen"] = metadata["first_seen_time"]
                person_info["last_seen"] = metadata["last_seen_time"]
            
            # Add coordinates if available
            if "coordinates" in metadata:
                person_info["coordinates"] = metadata["coordinates"]
        
        sorted_persons.append(person_info)
    
    # Sort by appearances (descending) for consistent "most appearances" responses
    sorted_persons.sort(key=lambda x: x.get("appearances", 0), reverse=True)
    processed_data["persons"] = sorted_persons
    
    # Create a summary of top persons by appearances
    processed_data["top_persons"] = []
    for i, person in enumerate(sorted_persons[:5]):  # Top 5 persons
        processed_data["top_persons"].append({
            "rank": i+1,
            "id": person.get("id"),
            "appearances": person.get("appearances", 0),
            "first_seen": person.get("first_seen", "unknown"),
            "last_seen": person.get("last_seen", "unknown")
        })
    
    # Calculate statistics for advanced queries
    appearances_list = [p.get("appearances", 0) for p in sorted_persons if p.get("appearances") is not None]
    if appearances_list:
        processed_data["max_appearances"] = max(appearances_list)
        processed_data["min_appearances"] = min(appearances_list)
        processed_data["avg_appearances"] = sum(appearances_list) / len(appearances_list)
        processed_data["persons_above_100"] = sum(1 for a in appearances_list if a > 100)
    
    # Process object counts
    if "object_counts" in case_data:
        processed_data["objects"] = case_data["object_counts"]
    
    # Process timeline
    if "timeline" in case_data:
        timeline = case_data["timeline"]
        processed_data["event_count"] = len(timeline)
        processed_data["events"] = []
        
        # Include all events for RAG retrieval
        for event in timeline:
            event_info = {
                "type": event.get("event_type", "Unknown"),
                "time": event.get("timestamp", "Unknown"),
                "description": event.get("description", "No description"),
                "frame": event.get("frame", "Unknown")
            }
            processed_data["events"].append(event_info)
    
    # Process frame data if available (for specific time-based queries)
    processed_data["frames"] = []
    if "frames" in case_data:
        # Just keep basic frame info to save tokens
        frame_count = len(case_data["frames"])
        processed_data["frame_count"] = frame_count
        
        # Sample some frames if there are too many
        if frame_count > 0:
            frame_keys = list(case_data["frames"].keys())
            sample_size = min(10, frame_count)
            sample_indices = [int(i * (frame_count / sample_size)) for i in range(sample_size)]
            
            for idx in sample_indices:
                if idx < len(frame_keys):
                    frame_key = frame_keys[idx]
                    frame = case_data["frames"][frame_key]
                    processed_data["frames"].append({
                        "frame_number": frame.get("frame_number", 0),
                        "timestamp": frame.get("timestamp", 0),
                        "detection_count": len(frame.get("detections", []))
                    })
    
    return processed_data

def retrieve_relevant_context(processed_data: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    """
    Implement RAG by retrieving relevant information from case data
    based on the user's query.
    """
    query_lower = query.lower()
    relevant_items = []
    
    # Check if query is about ranking or comparing persons
    ranking_query = False
    if any(term in query_lower for term in ["most", "least", "highest", "lowest", "more than", "less than", "top", "ranking"]):
        ranking_query = True
        
        # Check for specific ranking query types
        if any(term in query_lower for term in ["most", "highest", "top"]):
            top_persons = processed_data.get("top_persons", [])
            if top_persons:
                top_person = top_persons[0]  # The person with most appearances
                relevant_items.append({
                    "type": "top_person",
                    "content": f"The person with the most appearances is Person #{top_person.get('id')}, who appears {top_person.get('appearances')} times in the video. They were first seen at {top_person.get('first_seen')} and last seen at {top_person.get('last_seen')}."
                })
                
                # Add information about other top persons for context
                if len(top_persons) > 1:
                    top_persons_info = []
                    for i, person in enumerate(top_persons[:3]):  # Top 3
                        top_persons_info.append(f"Person #{person.get('id')} appears {person.get('appearances')} times")
                    
                    relevant_items.append({
                        "type": "top_persons",
                        "content": f"The top 3 persons by appearance count are: {', '.join(top_persons_info)}."
                    })
        
        # Check for threshold queries (more than X appearances)
        threshold_match = re.search(r'(more|greater|higher|over|above) than (\d+)', query_lower)
        if threshold_match:
            try:
                threshold = int(threshold_match.group(2))
                count_above = sum(1 for p in processed_data.get("persons", []) if p.get("appearances", 0) > threshold)
                
                # Get examples of persons above threshold
                examples = []
                for person in processed_data.get("persons", []):
                    if person.get("appearances", 0) > threshold:
                        examples.append(f"Person #{person.get('id')} ({person.get('appearances')} appearances)")
                        if len(examples) >= 3:  # Limit to 3 examples
                            break
                
                if count_above > 0:
                    relevant_items.append({
                        "type": "threshold_query",
                        "content": f"There are {count_above} persons with more than {threshold} appearances. Examples include: {', '.join(examples)}."
                    })
                else:
                    relevant_items.append({
                        "type": "threshold_query",
                        "content": f"No persons have more than {threshold} appearances in this video."
                    })
            except (ValueError, AttributeError):
                pass
    
    # Check if query is about a specific person
    person_match = re.search(r'person #?(\d+)', query_lower)
    if person_match:
        person_id = person_match.group(1)
        # Find the specific person
        for person in processed_data.get("persons", []):
            if str(person.get("id", "")) == person_id:
                relevant_items.append({
                    "type": "person",
                    "content": f"Person #{person_id} appears {person.get('appearances', 'unknown')} times in the video. First seen at {person.get('first_seen', 'unknown')} and last seen at {person.get('last_seen', 'unknown')}."
                })
                break
    
    # Check if query is about person count
    if any(term in query_lower for term in ["how many people", "how many persons", "number of people", "total people", "person count", "people detected"]):
        relevant_items.append({
            "type": "person_count",
            "content": f"There are {processed_data.get('person_count', 0)} persons detected in this video case."
        })
        
        # Add distribution information
        if processed_data.get("max_appearances") is not None:
            relevant_items.append({
                "type": "appearance_stats",
                "content": f"The person with the most appearances was detected {processed_data.get('max_appearances')} times, while the average number of appearances per person is {processed_data.get('avg_appearances', 0):.1f}."
            })
    
    # Check if query is about timeline or events
    if any(term in query_lower for term in ["timeline", "event", "when", "time", "happen"]):
        events = processed_data.get("events", [])
        if events:
            for event in events[:5]:  # Limit to 5 most relevant events
                relevant_items.append({
                    "type": "event",
                    "content": f"Event: {event.get('description', 'Unknown')} at time {event.get('time', 'unknown')} (frame {event.get('frame', 'unknown')})."
                })
        else:
            relevant_items.append({
                "type": "event_missing",
                "content": f"This case has {processed_data.get('person_count', 0)} persons detected, but no specific timeline events were recorded."
            })
    
    # Check if query is about object counts
    if any(term in query_lower for term in ["how many objects", "object count", "objects", "detected", "identify"]):
        if "objects" in processed_data and processed_data["objects"]:
            objects_str = ", ".join([f"{count} {obj}" for obj, count in processed_data.get("objects", {}).items()])
            relevant_items.append({
                "type": "objects",
                "content": f"Objects detected: {objects_str}."
            })
        
        relevant_items.append({
            "type": "counts",
            "content": f"Total persons detected: {processed_data.get('person_count', 0)}."
        })
    
    # General person-related questions
    if (any(term in query_lower for term in ["person", "people", "individual", "who", "someone"]) 
            and not ranking_query 
            and not person_match
            and not any(term in query_lower for term in ["how many", "count", "total"])):
        if processed_data.get("person_count", 0) > 0:
            relevant_items.append({
                "type": "person_summary",
                "content": f"There are {processed_data.get('person_count', 0)} persons detected in the video."
            })
            
            # Add information about a few persons
            for i, person in enumerate(processed_data.get("persons", [])[:3]):  # First 3 persons
                person_id = person.get("id", "")
                appearances = person.get("appearances", "unknown number of")
                first_seen = person.get("first_seen", "unknown time")
                last_seen = person.get("last_seen", "unknown time")
                
                relevant_items.append({
                    "type": "person_example",
                    "content": f"Person #{person_id} appears {appearances} times, first seen at {first_seen} and last seen at {last_seen}."
                })
    
    # If nothing specific found, add general case information
    if not relevant_items:
        relevant_items.append({
            "type": "case_summary",
            "content": f"Case {processed_data.get('case_id', 'Unknown')} contains video {processed_data.get('video_name', 'Unknown')} processed on {processed_data.get('timestamp', 'Unknown')}. It has {processed_data.get('person_count', 0)} persons detected."
        })
        
        # Add a sample of persons if available
        if processed_data.get("person_count", 0) > 0:
            sample_persons = processed_data.get("persons", [])[:3]  # Just the first 3
            for person in sample_persons:
                relevant_items.append({
                    "type": "person_sample",
                    "content": f"Person #{person.get('id', 'Unknown')} appears {person.get('appearances', 'unknown')} times in the video."
                })
    
    return relevant_items

def create_nshot_examples(processed_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Create dynamic n-shot examples based on the case data."""
    examples = []
    
    # Example 1: Person count
    examples.append({
        "question": "How many people are in this video?",
        "answer": f"There are {processed_data.get('person_count', 0)} persons detected in this video case. The analysis tracked each individual across multiple frames."
    })
    
    # Example 2: Timeline if available
    if processed_data.get("event_count", 0) > 0 and len(processed_data.get("events", [])) > 0:
        event = processed_data["events"][0]
        examples.append({
            "question": "What events happen in the timeline?",
            "answer": f"The case contains {processed_data.get('event_count', 0)} timeline events. One example is '{event.get('description', 'an event')}' which occurs at {event.get('time', 'unknown time')}."
        })
    else:
        examples.append({
            "question": "What events are in the timeline?",
            "answer": f"This case doesn't have specific timeline events recorded. However, I can tell you about the {processed_data.get('person_count', 0)} persons detected in the video, including when they first appear and when they're last seen."
        })
    
    # Example 3: Person with most appearances
    if processed_data.get("top_persons") and len(processed_data.get("top_persons", [])) > 0:
        top_person = processed_data["top_persons"][0]
        examples.append({
            "question": "Who appears the most in the video?",
            "answer": f"Person #{top_person.get('id')} appears the most, with {top_person.get('appearances')} appearances in the video. They were first seen at {top_person.get('first_seen')} and last seen at {top_person.get('last_seen')}."
        })
    
    # Example 4: Specific person
    if processed_data.get("persons") and len(processed_data.get("persons", [])) > 0:
        person = processed_data["persons"][0]
        person_id = person.get("id", "1")
        examples.append({
            "question": f"Tell me about Person #{person_id}",
            "answer": f"Person #{person_id} appears {person.get('appearances')} times in the video. They were first seen at {person.get('first_seen', 'unknown time')} and last seen at {person.get('last_seen', 'unknown time')}."
        })
    
    # Example 5: Threshold query
    examples.append({
        "question": "Does anyone appear more than 100 times?",
        "answer": f"Yes, {processed_data.get('persons_above_100', 0)} persons appear more than 100 times in the video. " + 
        (f"For example, Person #{processed_data['top_persons'][0]['id']} appears {processed_data['top_persons'][0]['appearances']} times." 
         if processed_data.get('top_persons') and processed_data['top_persons'][0].get('appearances', 0) > 100 else 
         "The person with the most appearances was detected {processed_data.get('max_appearances', 0)} times.")
    })
    
    # Example 6: General redirect for non-case questions
    examples.append({
        "question": "What is 2+2?",
        "answer": f"I'm focused on analyzing the video evidence for case {processed_data.get('case_id', 'Unknown')}. The case has {processed_data.get('person_count', 0)} persons detected. If you'd like to know more about the video analysis, please ask about the detected persons or timeline events."
    })
    
    return examples

def create_prompt(processed_data: Dict[str, Any], user_message: str, relevant_context: List[Dict[str, Any]]) -> str:
    """Create a well-structured prompt with n-shot examples and RAG context."""
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a summarized version of the case data
    case_summary = [
        f"Case ID: {processed_data.get('case_id', 'Unknown')}",
        f"Video: {processed_data.get('video_name', 'Unknown')}",
        f"Timestamp: {processed_data.get('timestamp', 'Unknown')}",
        f"Total persons detected: {processed_data.get('person_count', 0)}",
    ]
    
    # Add statistics about persons
    if processed_data.get("top_persons"):
        top_persons = processed_data["top_persons"]
        stats = [
            f"- Person with most appearances: Person #{top_persons[0]['id']} ({top_persons[0]['appearances']} appearances)",
            f"- Maximum appearances for any person: {processed_data.get('max_appearances', 0)}",
            f"- Average appearances per person: {processed_data.get('avg_appearances', 0):.1f}",
            f"- Persons with more than 100 appearances: {processed_data.get('persons_above_100', 0)}"
        ]
        case_summary.extend(stats)
    
    # Add object information if available
    if "objects" in processed_data and processed_data["objects"]:
        objects_str = ", ".join([f"{count} {obj}" for obj, count in processed_data.get("objects", {}).items()])
        if objects_str:
            case_summary.append(f"Objects detected: {objects_str}")
    
    # Add event information if available
    if "event_count" in processed_data:
        case_summary.append(f"Total events: {processed_data.get('event_count', 0)}")
    
    # Create n-shot examples
    examples = create_nshot_examples(processed_data)
    examples_text = "\n\n".join([
        f"Example Q: {ex['question']}\nExample A: {ex['answer']}" 
        for ex in examples
    ])
    
    # Add retrieved context - prioritize by relevance type
    context_items = []
    context_types_seen = set()
    
    # First, add specific person information or ranking information if the query is about that
    for item in relevant_context:
        item_type = item.get("type", "")
        if item_type in ["person", "top_person", "top_persons", "threshold_query"] and item_type not in context_types_seen:
            context_items.append(item["content"])
            context_types_seen.add(item_type)
    
    # Next, add person count and statistics
    for item in relevant_context:
        item_type = item.get("type", "")
        if item_type in ["person_count", "appearance_stats"] and item_type not in context_types_seen:
            context_items.append(item["content"])
            context_types_seen.add(item_type)
    
    # Finally, add any other context
    for item in relevant_context:
        item_type = item.get("type", "")
        if item_type not in context_types_seen:
            context_items.append(item["content"])
            context_types_seen.add(item_type)
    
    context_text = "\n".join(context_items)
    
    # Construct the prompt with clear instructions + n-shot + RAG
    prompt = f"""You are VideoProof AI, a highly accurate and helpful assistant analyzing video evidence. Current date: {current_date}. Answer questions about this case data:

CASE SUMMARY:
{chr(10).join(case_summary)}

RELEVANT CONTEXT FOR THIS QUESTION:
{context_text}

PREVIOUS EXAMPLES OF GOOD RESPONSES:
{examples_text}

USER QUESTION:
{user_message}

Provide a detailed, accurate answer based only on the case data. Be specific and direct - answer exactly what was asked. If data shows multiple persons, always include specific details like appearance counts. Never contradict yourself. If you don't have sufficient information, acknowledge that politely and offer what you do know.
"""
    
    return prompt

def get_demo_response(processed_data: Dict[str, Any], user_message: str, relevant_context: List[Dict[str, Any]]) -> str:
    """Generate informative responses in demo mode using the RAG context."""
    question = user_message.lower()
    
    # Use the retrieved context to build a comprehensive response
    context_items = []
    for item in relevant_context:
        context_items.append(item["content"])
    
    context_text = " ".join(context_items)
    
    # Handle person count questions
    if any(term in question for term in ["how many people", "how many persons", "person count", "number of people", "people detected"]):
        count = processed_data.get("person_count", 0)
        response = f"There are {count} persons detected in this video case."
        
        # Add statistics if available
        if processed_data.get("max_appearances") is not None:
            response += f" The person with the most appearances was detected {processed_data.get('max_appearances')} times, and the average number of appearances per person is {processed_data.get('avg_appearances', 0):.1f}."
        
        return response
    
    # Handle "who appears the most" questions
    if any(term in question for term in ["most", "highest", "top"]) and any(term in question for term in ["appear", "detected", "seen"]):
        if processed_data.get("top_persons"):
            top_persons = processed_data["top_persons"]
            response = f"Person #{top_persons[0]['id']} appears the most, with {top_persons[0]['appearances']} appearances in the video. They were first seen at {top_persons[0]['first_seen']} and last seen at {top_persons[0]['last_seen']}."
            
            # Add information about other top persons
            if len(top_persons) > 1:
                response += f" The next most frequent are: "
                other_persons = []
                for i in range(1, min(3, len(top_persons))):
                    other_persons.append(f"Person #{top_persons[i]['id']} with {top_persons[i]['appearances']} appearances")
                response += ", ".join(other_persons) + "."
            
            return response
        else:
            return "I don't have detailed information about person appearance frequency in this case."
    
    # Handle threshold queries (more than X appearances)
    threshold_match = re.search(r'(more|greater|higher|over|above) than (\d+)', question)
    if threshold_match:
        try:
            threshold = int(threshold_match.group(2))
            above_threshold = [p for p in processed_data.get("persons", []) if p.get("appearances", 0) > threshold]
            count_above = len(above_threshold)
            
            if count_above > 0:
                response = f"Yes, {count_above} persons appear more than {threshold} times in the video."
                
                # Add examples
                if above_threshold:
                    response += " These include: "
                    examples = []
                    for i, person in enumerate(above_threshold[:3]):  # Limit to 3 examples
                        examples.append(f"Person #{person['id']} with {person['appearances']} appearances")
                    response += ", ".join(examples) + "."
                
                return response
            else:
                return f"No, none of the {processed_data.get('person_count', 0)} persons detected appear more than {threshold} times in the video. The maximum number of appearances for any person is {processed_data.get('max_appearances', 0)}."
        except:
            pass
    
    # Handle questions about specific persons
    person_match = re.search(r'person #?(\d+)', question)
    if person_match:
        person_id = person_match.group(1)
        for person in processed_data.get("persons", []):
            if str(person.get("id", "")) == person_id:
                return f"Person #{person_id} appears {person.get('appearances', 0)} times in the video. They were first seen at {person.get('first_seen', 'unknown time')} and last seen at {person.get('last_seen', 'unknown time')}."
        
        return f"I don't have specific information about Person #{person_id} in my data. The case has {processed_data.get('person_count', 0)} persons in total, with IDs ranging from 1 to {processed_data.get('person_count', 0)}."
    
    # Handle timeline questions
    if any(term in question for term in ["timeline", "events", "when", "what happened"]):
        events = processed_data.get("events", [])
        if events:
            event_text = "\n- ".join([f"{e.get('description', 'Event')} at {e.get('time', 'unknown time')}" for e in events[:3]])
            return f"The case timeline contains {processed_data.get('event_count', 0)} events. Here are some key events:\n- {event_text}"
        else:
            return f"This case doesn't contain detailed timeline events. However, I can tell you there are {processed_data.get('person_count', 0)} persons detected throughout the video."
    
    # Handle object questions
    if any(term in question for term in ["object", "detect", "found", "identify"]) and not any(term in question for term in ["person", "people"]):
        if "objects" in processed_data and processed_data["objects"]:
            objects_str = ", ".join([f"{count} {obj}" for obj, count in processed_data.get("objects", {}).items()])
            return f"The analysis detected the following objects in the video: {objects_str}."
        else:
            return f"This case primarily tracks {processed_data.get('person_count', 0)} persons in the video, but doesn't contain specific counts of other object types."
    
    # Handle math or non-case questions
    if any(term in question for term in ["+", "plus", "add", "sum", "multiply", "divide", "calculate"]):
        return "I'm designed to answer questions about the video case data. For this case, I can tell you there are " + \
               f"{processed_data.get('person_count', 0)} persons detected in the video. Please ask me about the case details if you'd like to know more."
    
    # Default response with helpful information
    if context_text:
        return context_text
    else:
        return f"This case contains video evidence with {processed_data.get('person_count', 0)} persons detected. You can ask about specific persons, their appearance counts, or who appears most frequently in the video."

def generate_chat_response(case_id: str, case_data: Dict[str, Any], user_message: str) -> str:
    """Generate a helpful response to a user's question about case data."""
    try:
        # Make sure case_id is in the data
        if 'case_id' not in case_data:
            case_data['case_id'] = case_id
            
        # Preprocess the case data for cleaner prompting
        processed_data = preprocess_case_data(case_data)
        
        # Retrieve relevant context (RAG approach)
        relevant_context = retrieve_relevant_context(processed_data, user_message)
        
        # Import here to avoid potential circular imports
        import sys
        current_module = sys.modules[__name__]
        if not hasattr(current_module, 'hf_client') or not hasattr(current_module, 'MODEL_ID'):
            # Import from app.py if we don't have these attributes
            try:
                from app import hf_client, MODEL_ID
            except ImportError:
                hf_client = None
                MODEL_ID = None
        else:
            hf_client = current_module.hf_client
            MODEL_ID = current_module.MODEL_ID
        
        # If no HF client available, use demo mode
        if not hf_client:
            return get_demo_response(processed_data, user_message, relevant_context)
        
        # Create the prompt with n-shot examples and RAG context
        prompt = create_prompt(processed_data, user_message, relevant_context)
        
        # Generate response using the Hugging Face API
        try:
            response = hf_client.text_generation(
                prompt,
                model=MODEL_ID,
                max_new_tokens=250,  # Respect the 250 token limit
                temperature=0.7,
                repetition_penalty=1.2
            )
            
            # Clean up response if needed
            cleaned_response = response.strip()
            
            # If we got an empty or very short response, fall back to demo mode
            if len(cleaned_response) < 10:
                logger.warning(f"Got very short response from API: '{cleaned_response}'. Using fallback.")
                return get_demo_response(processed_data, user_message, relevant_context)
                
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Error calling Hugging Face API: {e}")
            # Fall back to demo mode
            return get_demo_response(processed_data, user_message, relevant_context)
    
    except Exception as e:
        logger.error(f"Error generating chat response: {e}")
        return f"I encountered an error analyzing the case data. This case has approximately {len(case_data.get('person_identities', []))} persons detected. You can ask specific questions about them or the video timeline."