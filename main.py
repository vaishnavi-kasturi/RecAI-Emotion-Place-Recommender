import os
import threading
import time
import webbrowser
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
import requests
import pandas as pd
import uvicorn
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict
import logging
from math import sqrt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# 1. FastAPI app
# -------------------------------
app = FastAPI(title="Places Emotion Recommender API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 2. MongoDB setup
# -------------------------------
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["places_db"]
places_collection = db["nearby_places"]

# -------------------------------
# 3. Enhanced Place Information Fetcher
# -------------------------------
def get_place_details_from_overpass(place_id: str, place_type: str) -> Dict[str, str]:
    """
    Fetch detailed information about a place using Overpass API
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Query for detailed information
    query = f"""
[out:json][timeout:15];
{place_type}({place_id});
out tags;
"""
    
    try:
        response = requests.get(overpass_url, params={"data": query}, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        if data.get("elements"):
            tags = data["elements"][0].get("tags", {})
            
            # Extract description from various tag sources
            description_parts = []
            
            # Primary description sources
            if tags.get("description"):
                description_parts.append(tags["description"])
            if tags.get("tourism:description"):
                description_parts.append(tags["tourism:description"])
            if tags.get("note"):
                description_parts.append(tags["note"])
                
            # Build contextual description from tags
            context_parts = []
            
            # Cuisine information
            if tags.get("cuisine"):
                context_parts.append(f"serves {tags['cuisine']} cuisine")
            
            # Service/facility information
            if tags.get("amenity"):
                amenity = tags["amenity"]
                if amenity == "restaurant":
                    context_parts.append("dining establishment")
                elif amenity == "cafe":
                    context_parts.append("coffee shop and casual dining")
                elif amenity == "hospital":
                    context_parts.append("medical care facility")
                elif amenity == "bank":
                    context_parts.append("financial services")
                elif amenity == "pharmacy":
                    context_parts.append("medication and health products")
                elif amenity == "school":
                    context_parts.append("educational institution")
                elif amenity == "library":
                    context_parts.append("books and study facility")
                elif amenity == "gym":
                    context_parts.append("fitness and exercise facility")
                else:
                    context_parts.append(f"{amenity} facility")
            
            # Tourism information
            if tags.get("tourism"):
                tourism = tags["tourism"]
                if tourism == "hotel":
                    context_parts.append("accommodation and lodging")
                elif tourism == "museum":
                    context_parts.append("cultural exhibitions and artifacts")
                elif tourism == "attraction":
                    context_parts.append("tourist destination and sightseeing")
                elif tourism == "viewpoint":
                    context_parts.append("scenic overlook with views")
                else:
                    context_parts.append(f"{tourism} destination")
            
            # Leisure information
            if tags.get("leisure"):
                leisure = tags["leisure"]
                if leisure == "park":
                    context_parts.append("outdoor recreation and nature")
                elif leisure == "sports_centre":
                    context_parts.append("sports activities and fitness")
                elif leisure == "swimming_pool":
                    context_parts.append("swimming and water activities")
                elif leisure == "garden":
                    context_parts.append("landscaped outdoor space")
                else:
                    context_parts.append(f"{leisure} activity")
            
            # Shop information
            if tags.get("shop"):
                shop = tags["shop"]
                if shop == "mall":
                    context_parts.append("shopping center with multiple stores")
                elif shop == "supermarket":
                    context_parts.append("grocery and daily necessities")
                elif shop == "clothes":
                    context_parts.append("clothing and fashion retail")
                elif shop == "book":
                    context_parts.append("books and reading materials")
                else:
                    context_parts.append(f"{shop} retail store")
            
            # Additional contextual information
            if tags.get("building"):
                building_type = tags["building"]
                if building_type in ["church", "temple", "mosque", "synagogue"]:
                    context_parts.append("place of worship and spiritual activities")
                elif building_type == "hospital":
                    context_parts.append("medical treatment and healthcare")
                elif building_type == "school":
                    context_parts.append("learning and educational programs")
            
            # Combine all description parts
            full_description = " ".join(description_parts)
            if context_parts:
                contextual_info = ", ".join(context_parts)
                if full_description:
                    full_description += f". {contextual_info}"
                else:
                    full_description = contextual_info
            
            return {
                "description": full_description,
                "raw_tags": tags
            }
    except Exception as e:
        print(f"⚠️ Could not fetch detailed info: {e}")
        return {"description": "", "raw_tags": {}}
    
    return {"description": "", "raw_tags": {}}

# -------------------------------
# 4. Enhanced Emotion Analysis for Places
# -------------------------------
class PlaceEmotionAnalyzer:
    def __init__(self):
        self.model = None
        self.reference_data = None
        self._load_model()
        self._load_reference_data()
    
    def _load_model(self):
        try:
            # Using a more context-aware model for better understanding
            self.model = SentenceTransformer("all-mpnet-base-v2")
            print("✅ Loaded context-aware emotion analysis model successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _load_reference_data(self):
        file_path = r"C:\Users\rhythm\Desktop\recsys\data\df_result.xlsx"
        
        try:
            if os.path.exists(file_path):
                df_result = pd.read_excel(file_path)
                df_result = df_result.dropna(subset=["category_combined", "place_emotion"])
                
                if "place_description" in df_result.columns:
                    df_result['text_for_embedding'] = (
                        df_result['category_combined'].astype(str) + " " + 
                        df_result['place_description'].astype(str)
                    )
                else:
                    df_result['text_for_embedding'] = df_result['category_combined'].astype(str)
                
                print("🔄 Computing embeddings for reference data...")
                self.reference_embeddings = self.model.encode(
                    df_result['text_for_embedding'].tolist(), 
                    convert_to_tensor=True
                )
                self.reference_data = df_result
                print(f"✅ Loaded {len(df_result)} reference places with emotions")
            else:
                print("⚠️ Reference dataset not found, using enhanced emotion mapping")
                self.reference_data = None
        except Exception as e:
            print(f"❌ Error loading reference data: {e}")
            self.reference_data = None
    
    def create_context_aware_text(self, place_name: str, place_category: str, place_description: str = None, tags: Dict = None) -> str:
        """
        Create a rich, context-aware text representation of the place
        """
        # Start with name and category
        context_text = f"{place_name} is a {place_category}"
        
        # Add description if available
        if place_description and place_description.strip():
            context_text += f". {place_description}"
        
        # Add contextual information from tags
        if tags:
            context_additions = []
            
            # Add opening hours context
            if tags.get("opening_hours"):
                context_additions.append("operates with specific hours")
            
            # Add accessibility info
            if tags.get("wheelchair") == "yes":
                context_additions.append("wheelchair accessible")
            
            # Add atmosphere indicators
            if tags.get("outdoor_seating") == "yes":
                context_additions.append("offers outdoor seating")
            if tags.get("takeaway") == "yes":
                context_additions.append("provides takeaway service")
            if tags.get("wifi") == "yes":
                context_additions.append("has wifi connectivity")
            
            # Add price indicators
            if tags.get("price_range"):
                price = tags["price_range"]
                if price in ["$", "cheap"]:
                    context_additions.append("budget-friendly pricing")
                elif price in ["$$", "moderate"]:
                    context_additions.append("moderate pricing")
                elif price in ["$$$", "$$$$", "expensive"]:
                    context_additions.append("upscale pricing")
            
            # Add brand/chain context
            if tags.get("brand"):
                context_additions.append(f"part of {tags['brand']} chain")
            
            if context_additions:
                context_text += f". Features include: {', '.join(context_additions)}"
        
        return context_text
    
    def predict_emotions_for_place(self, place_name: str, place_category: str, 
                                   place_description: str = None, tags: Dict = None) -> List[Dict]:
        """
        Enhanced emotion prediction using context-aware text analysis
        """
        # Create rich context-aware text
        context_aware_text = self.create_context_aware_text(
            place_name, place_category, place_description, tags
        )
        
        print(f"🧠 Context text for {place_name}: {context_aware_text[:100]}...")
        
        try:
            if self.reference_data is not None:
                return self._get_emotions_from_similarity(context_aware_text)
            else:
                return self._get_enhanced_emotions_from_context(context_aware_text, place_category, tags)
        except Exception as e:
            print(f"❌ Error predicting emotions for {place_name}: {e}")
            return [{"emotion": "neutral", "confidence": 0.5}]
    
    def _get_emotions_from_similarity(self, context_text: str, top_k: int = 3) -> List[Dict]:
        """
        Use semantic similarity with reference data for emotion prediction
        """
        place_embedding = self.model.encode(context_text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(place_embedding, self.reference_embeddings)[0]
        top_results = torch.topk(cosine_scores, k=min(top_k * 3, len(cosine_scores)))
        
        emotion_scores = {}
        for score, idx in zip(top_results.values, top_results.indices):
            if score.item() > 0.2:  # Lower threshold for more nuanced matching
                emotion = self.reference_data.iloc[idx.item()]['place_emotion']
                confidence = float(score.item())
                if emotion in emotion_scores:
                    emotion_scores[emotion] = max(emotion_scores[emotion], confidence)
                else:
                    emotion_scores[emotion] = confidence
        
        emotions = []
        for emotion, confidence in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            emotions.append({"emotion": emotion, "confidence": round(confidence, 3)})
        
        return emotions if emotions else [{"emotion": "neutral", "confidence": 0.5}]
    def _get_enhanced_emotions_from_context(self, context_text: str, category: str, tags: Dict = None) -> List[Dict]:
        emotions = []
        context_lower = context_text.lower()
        if any(word in context_lower for word in ["temple", "church", "mosque", "worship", "spiritual", "shrine", "monastery", "cathedral"]):
            emotions.append({"emotion": "spirituality", "confidence": 0.9})  # Changed from "peaceful" to "spirituality"
            emotions.append({"emotion": "contemplative", "confidence": 0.8})
            emotions.append({"emotion": "peaceful", "confidence": 0.7})  # Keep peaceful as secondary
        if any(word in context_lower for word in ["park", "garden", "outdoor", "nature", "scenic", "viewpoint", "spa", "wellness"]):
            emotions.append({"emotion": "calm", "confidence": 0.9})  # Added explicit "calm"
            emotions.append({"emotion": "peaceful", "confidence": 0.8})
            emotions.append({"emotion": "relaxed", "confidence": 0.8})
                # Food and dining emotions
        if any(word in context_lower for word in ["restaurant", "cafe", "dining", "food", "cuisine"]):
            emotions.append({"emotion": "social", "confidence": 0.8})
            emotions.append({"emotion": "comfort", "confidence": 0.7})
            if "upscale" in context_lower or "fine dining" in context_lower:
                emotions.append({"emotion": "luxury", "confidence": 0.8})
    
    # Cultural and educational emotions
        if any(word in context_lower for word in ["museum", "cultural", "exhibition", "library", "educational"]):
            emotions.append({"emotion": "curious", "confidence": 0.8})
            emotions.append({"emotion": "contemplative", "confidence": 0.7})
    
    # Fitness and health emotions
        if any(word in context_lower for word in ["gym", "fitness", "sports", "exercise", "swimming"]):
            emotions.append({"emotion": "energetic", "confidence": 0.9})
            emotions.append({"emotion": "confident", "confidence": 0.7})
    
    # Shopping emotions
        if any(word in context_lower for word in ["shop", "mall", "retail", "store"]):
            emotions.append({"emotion": "excitement", "confidence": 0.7})
            emotions.append({"emotion": "social", "confidence": 0.6})
    
    # Entertainment emotions
        if any(word in context_lower for word in ["cinema", "theater", "entertainment", "attraction"]):
            emotions.append({"emotion": "excitement", "confidence": 0.8})
            emotions.append({"emotion": "joy", "confidence": 0.7})
    
    # Accommodation emotions
        if any(word in context_lower for word in ["hotel", "accommodation", "lodging"]):
            emotions.append({"emotion": "comfort", "confidence": 0.8})
            emotions.append({"emotion": "relaxed", "confidence": 0.7})
    
    # Default fallback
        if not emotions:
            emotions = [{"emotion": "neutral", "confidence": 0.5}]
    
    # Limit to top 3 emotions
        return sorted(emotions, key=lambda x: x["confidence"], reverse=True)[:3]


emotion_analyzer = PlaceEmotionAnalyzer()

# -------------------------------
# 5. Input models
# -------------------------------
class Location(BaseModel):
    latitude: float
    longitude: float

class UserPreference(BaseModel):
    latitude: float
    longitude: float
    emotions: List[str]  # Changed to match frontend format
    vector: List[int] = None  # Optional binary vector
    timestamp: str = None  # Optional timestamp

# -------------------------------
# 6. Enhanced fetch nearby places with detailed context
# -------------------------------
def get_nearby_places_with_emotions(lat, lon, radius=10000):
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Enhanced query to get more detailed information
    query = f"""
[out:json][timeout:30];
(
  node(around:{radius},{lat},{lon})[amenity][name];
  node(around:{radius},{lat},{lon})[tourism][name];
  node(around:{radius},{lat},{lon})[leisure][name];
  node(around:{radius},{lat},{lon})[shop][name];
  way(around:{radius},{lat},{lon})[amenity][name];
  way(around:{radius},{lat},{lon})[tourism][name];
  way(around:{radius},{lat},{lon})[leisure][name];
  way(around:{radius},{lat},{lon})[shop][name];
);
out center tags 100;
"""

    try:
        print("🔄 Fetching places from OpenStreetMap with detailed information...")
        response = requests.get(overpass_url, params={"data": query}, timeout=40)
        response.raise_for_status()
        data = response.json()
        print(f"📍 Found {len(data.get('elements', []))} raw places")
    except Exception as e:
        print(f"❌ Error fetching places: {e}")
        return []
    
    places_with_emotions = []
    processed_names = set()
    
    for element in data.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name", "").strip()
        
        # Skip places without names or already processed
        if not name or name in processed_names:
            continue
        processed_names.add(name)
        
        # Get primary category
        category = (tags.get("amenity") or tags.get("tourism") or 
                    tags.get("leisure") or tags.get("shop") or "unknown")
        
        # Skip unknown categories
        if category.lower() == "unknown":
            continue
        
        # Get coordinates
        lat_val = element.get("lat") or element.get("center", {}).get("lat")
        lon_val = element.get("lon") or element.get("center", {}).get("lon")
        
        # Create initial description from tags
        initial_description = f"{name} - {category}"
        if tags.get("cuisine"):
            initial_description += f" ({tags.get('cuisine')})"
        
        # Get enhanced place details
        element_type = element.get("type", "node")
        element_id = element.get("id")
        
        print(f"📝 Fetching detailed info for: {name}")
        place_details = get_place_details_from_overpass(element_id, element_type)
        
        # Combine name with enhanced description
        enhanced_description = place_details.get("description", "")
        if not enhanced_description:
            enhanced_description = initial_description
        else:
            enhanced_description = f"{initial_description}. {enhanced_description}"
        
        # Create combined context text for emotion analysis
        print(f"🧠 Analyzing emotions for: {name}")
        emotions = emotion_analyzer.predict_emotions_for_place(
            name, category, enhanced_description, tags
        )
        
        emotion_vector = {e["emotion"]: e["confidence"] for e in emotions}
        
        place_data = {
            "name": name,
            "category": category,
            "description": enhanced_description,
            "combined_context": f"{name} {enhanced_description}",  # This is what we use for analysis
            "lat": lat_val,
            "lon": lon_val,
            "emotions": emotions,
            "emotion_vector": emotion_vector,
            "osm_tags": tags,
            "raw_details": place_details.get("raw_tags", {}),
            "created_at": pd.Timestamp.now().isoformat()
        }
        places_with_emotions.append(place_data)
    
    print(f"✅ Processed {len(places_with_emotions)} unique places with context-aware emotions")
    return places_with_emotions

# -------------------------------
# 7. API endpoints (keeping existing structure)
# -------------------------------
@app.post("/fetch_places")
def fetch_and_store_places_with_emotions(loc: Location):
    try:
        lat, lon = loc.latitude, loc.longitude
        print(f"🎯 Fetching places with enhanced context analysis for coordinates: {lat}, {lon}")
        places = get_nearby_places_with_emotions(lat, lon)
        if not places:
            print("⚠️ No places found nearby")
            return {"status": "success", "message": "No places found nearby", "total_places": 0}
        
        places_collection.delete_many({})
        result = places_collection.insert_many(places)
        print(f"💾 Stored {len(result.inserted_ids)} places in database")
        
        # Save to Excel with enhanced information
        try:
            df = pd.DataFrame(places)
            excel_path = os.path.join(os.getcwd(), "places_with_enhanced_emotions.xlsx")
            df.to_excel(excel_path, index=False)
            print(f"📊 Saved enhanced report to: {excel_path}")
        except Exception as e:
            print(f"⚠️ Could not save Excel file: {e}")
            excel_path = "N/A"
        
        return {
            "status": "success",
            "message": "Places fetched with context-aware emotion analysis",
            "total_places": len(places),
            "excel_file": excel_path
        }
    except Exception as e:
        print(f"❌ Error in fetch_and_store_places_with_emotions: {e}")
        return {"status": "error", "message": str(e), "total_places": 0}

@app.get("/places")
def get_all_places_with_emotions():
    try:
        places = list(places_collection.find({}, {"_id": 0}))
        if not places:
            return {"status": "success", "message": "No places found", "places": [], "count": 0}
        
        all_emotions = set()
        emotion_counts = {}
        for place in places:
            for e in place.get("emotions", []):
                all_emotions.add(e["emotion"])
                emotion_counts[e["emotion"]] = emotion_counts.get(e["emotion"], 0) + 1
        
        return {
            "status": "success",
            "places": places,
            "count": len(places),
            "emotion_summary": {"unique_emotions": list(all_emotions), "emotion_frequency": emotion_counts}
        }
    except Exception as e:
        print(f"❌ Error retrieving places: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test_emotion/{place_name}")
def test_single_emotion_analysis(place_name: str, category: str = "restaurant", description: str = ""):
    try:
        emotions = emotion_analyzer.predict_emotions_for_place(place_name, category, description)
        emotion_vector = {e["emotion"]: e["confidence"] for e in emotions}
        context_text = emotion_analyzer.create_context_aware_text(place_name, category, description)
        return {
            "place_name": place_name, 
            "category": category, 
            "description": description,
            "context_text": context_text,
            "predicted_emotions": emotions, 
            "emotion_vector": emotion_vector
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/debug_place/{place_name}")
def debug_place_matching(place_name: str, user_emotions: str = "Spirituality,Relaxation/Calm"):
    """
    Debug a specific place's emotion matching
    """
    try:
        # Parse user emotions
        emotions_list = [e.strip() for e in user_emotions.split(',')]
        
        # Find the place in database
        place = places_collection.find_one({"name": {"$regex": place_name, "$options": "i"}}, {"_id": 0})
        
        if not place:
            return {"error": f"Place '{place_name}' not found"}
        
        # Calculate match score
        match_info = calculate_emotion_match_score(emotions_list, place.get("emotion_vector", {}))
        
        return {
            "place_name": place.get("name"),
            "place_emotions": place.get("emotions", []),
            "emotion_vector": place.get("emotion_vector", {}),
            "user_emotions": emotions_list,
            "match_info": match_info,
            "would_pass_filtering": match_info["final_score"] > 0.01,
            "description": place.get("description", "")
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    places_count = places_collection.count_documents({})
    return {
        "status": "healthy",
        "emotion_model_loaded": emotion_analyzer.model is not None,
        "reference_data_loaded": emotion_analyzer.reference_data is not None,
        "places_in_database": places_count,
        "ready_for_recommendations": places_count > 0,
        "model_type": "Context-aware SentenceTransformer"
    }

# -------------------------------
# 8. Improved Recommendation System (keeping existing logic)
# -------------------------------
def calculate_emotion_match_score(user_emotions: List[str], place_emotion_vector: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate detailed matching scores between user emotions and place emotions
    Returns scores for exact matches, partial matches, and overall compatibility
    """
    # Map frontend emotion names to backend emotion names
    emotion_mapping = {
        "Joy/Happy": ["joy,happy", "joyful","ice_cream"],
        "Relaxation/Calm": ["relaxation,calm", "relaxed", "peaceful", "tranquil", "serene", "restaurant"],  # Added more mappings
        "Social Emotion": ["social","restaurent","beverages","drinks","cafe"],
        "Excitement": ["excitement", "energetic", "thrilling"],
        "Comfort": ["comfort", "cozy", "comfortable"],
        "Adventure": ["adventure", "exploration", "adventurous"],
        "Romance": ["romantic", "intimate"],
        "Luxury": ["luxury", "premium", "upscale"],
        "Shopping": ["shopping", "retail", "supermarket","clothes"],
        "Nostalgia": ["nostalgia", "nostalgic"],
        "Energy": ["energetic", "energy", "active"],
        "Wellness": ["wellness", "health"],
        "Entertainment": ["entertainment"],
        "Exploration": ["exploration", "curious", "discovery", "atrraction","museum"],
        "Fear": ["fear", "anxiety"],
        "Creativity": ["creative", "artistic"],
        "Spirituality": ["spiritual", "contemplative", "peaceful", "place_of_worship", "spirituality"],  # FIXED: Added "spirituality"
        "Education": ["educational", "learning", "curious"],
        "Retail": ["retail", "shopping"],
        "Outdoors": ["outdoors","excitement", "nature", "peaceful","summer_camp","sports_centre","cafe"]
    }
    
    # Flatten user emotions to backend emotion names
    user_backend_emotions = []
    for emotion in user_emotions:
        mapped = emotion_mapping.get(emotion, [emotion.lower()])
        user_backend_emotions.extend(mapped)
    
    # Remove neutral emotions from place vector for scoring
    filtered_place_vector = {k: v for k, v in place_emotion_vector.items() 
                           if k != "neutral" and v > 0.2}
    
    if not filtered_place_vector:
        return {
            "exact_matches": 0,
            "exact_match_score": 0.0,
            "weighted_score": 0.0,
            "coverage_score": 0.0,
            "final_score": 0.0,
            "matched_emotions": []
        }
    
    # Calculate exact matches
    exact_matches = []
    exact_match_score = 0.0
    
    for user_emotion in user_backend_emotions:
        if user_emotion in filtered_place_vector:
            confidence = filtered_place_vector[user_emotion]
            exact_matches.append((user_emotion, confidence))
            exact_match_score += confidence
    
    # Calculate weighted score (prefers places with multiple matching emotions)
    weighted_score = exact_match_score * (1 + 0.5 * len(exact_matches))
    
    # Calculate coverage score (how many of user's emotions are covered)
    coverage_score = len(exact_matches) / len(set(user_backend_emotions)) if user_backend_emotions else 0
    
    # Calculate final score combining all factors
    final_score = (
        exact_match_score * 0.4 +
        weighted_score * 0.4 +
        coverage_score * 0.2
    )
    
    return {
        "exact_matches": len(exact_matches),
        "exact_match_score": round(exact_match_score, 3),
        "weighted_score": round(weighted_score, 3),
        "coverage_score": round(coverage_score, 3),
        "final_score": round(final_score, 3),
        "matched_emotions": exact_matches
    }

def rank_places_by_emotion_priority(places: List[Dict], user_emotions: List[str]) -> List[Dict]:
    """
    Rank places with priority system using enhanced context-aware analysis
    """
    scored_places = []
    
    print(f"🔍 DEBUG: Starting with {len(places)} places")
    print(f"🔍 DEBUG: User selected emotions: {user_emotions}")
    
    for place in places:
        emotion_vector = place.get("emotion_vector", {})
        place_name = place.get('name', 'Unknown')
        
        print(f"\n📍 Analyzing: {place_name}")
        print(f"   Emotion vector: {emotion_vector}")
        
        # Check if place has only neutral emotions
        non_neutral_emotions = {k: v for k, v in emotion_vector.items() if k != "neutral"}
        
        # FIXED: More lenient filtering - don't immediately discard neutral-heavy places
        if not non_neutral_emotions:
            print(f"🚫 Discarding {place_name} - Only neutral emotions")
            continue
        
        # Get match info to see if user emotions are present
        match_info = calculate_emotion_match_score(user_emotions, emotion_vector)
        
        print(f"   Match info: {match_info}")
        
        # FIXED: Much more lenient filtering
        neutral_confidence = emotion_vector.get("neutral", 0)
        max_non_neutral_confidence = max(non_neutral_emotions.values()) if non_neutral_emotions else 0
        has_user_emotion_match = match_info["exact_matches"] > 0
        
        # FIXED: Only filter out if overwhelmingly neutral AND no matches
        if neutral_confidence > 0.8 and not has_user_emotion_match:
            print(f"🚫 Discarding {place_name} - Overwhelmingly neutral with no emotion matches")
            continue
        
        # FIXED: Much lower threshold for emotion match scores
        if match_info["final_score"] <= 0.01:  # Changed from 0.05 to 0.01
            print(f"🚫 Discarding {place_name} - Very low emotion match (score: {match_info['final_score']})")
            continue
            
        place_with_score = {
            **place,
            "match_info": match_info,
            "final_score": match_info["final_score"]
        }
        scored_places.append(place_with_score)
        print(f"✅ Keeping: {place_name} - Score: {match_info['final_score']:.3f}, Matches: {match_info['exact_matches']}")
    
    print(f"\n🎯 Final results: {len(scored_places)} places after filtering")
    
    # Sort by final score (descending)
    return sorted(scored_places, key=lambda x: x["final_score"], reverse=True)



@app.post("/recommend_places")
def recommend_places(user: UserPreference, top_k: int = 20):
    try:
        print(f"🎯 Context-aware recommendation request for location: {user.latitude}, {user.longitude}")
        print(f"📝 Selected emotions: {user.emotions}")
        
        places = list(places_collection.find({}, {"_id": 0}))
        if not places:
            return {"status": "error", "message": "No places in database. Use /fetch_places first."}
        
        # Use improved ranking system with context-aware analysis
        recommendations = rank_places_by_emotion_priority(places, user.emotions)
        
        # Limit results
        recommendations = recommendations[:top_k]
        
        # Add debug information for first few results
        debug_info = []
        for i, rec in enumerate(recommendations[:5]):
            match_info = rec.get("match_info", {})
            debug_info.append({
                "rank": i + 1,
                "place_name": rec.get("name", "Unknown"),
                "category": rec.get("category", "Unknown"),
                "context_snippet": rec.get("combined_context", "")[:100] + "..." if len(rec.get("combined_context", "")) > 100 else rec.get("combined_context", ""),
                "final_score": match_info.get("final_score", 0),
                "exact_matches": match_info.get("exact_matches", 0),
                "matched_emotions": match_info.get("matched_emotions", []),
                "all_emotions": list(rec.get("emotion_vector", {}).keys())
            })
        
        print(f"✅ Generated {len(recommendations)} context-aware recommendations")
        print("🔍 Top 5 recommendations debug info:")
        for debug in debug_info:
            print(f"   {debug['rank']}. {debug['place_name']} (Score: {debug['final_score']}, Matches: {debug['exact_matches']})")
            print(f"      Context: {debug['context_snippet']}")
        
        # Clean up recommendations for response
        clean_recommendations = []
        for rec in recommendations:
            clean_rec = {k: v for k, v in rec.items() if k not in ["match_info"]}
            # Add summary of matching emotions for frontend
            match_info = rec.get("match_info", {})
            clean_rec["emotion_match_summary"] = {
                "score": match_info.get("final_score", 0),
                "matched_emotions": [emotion for emotion, confidence in match_info.get("matched_emotions", [])],
                "match_count": match_info.get("exact_matches", 0)
            }
            clean_recommendations.append(clean_rec)
        
        return {
            "status": "success",
            "user_location": {"lat": user.latitude, "lon": user.longitude},
            "user_emotions": user.emotions,
            "recommendations": clean_recommendations,
            "count": len(clean_recommendations),
            "analysis_type": "context-aware",
            "debug_info": debug_info
        }
    except Exception as e:
        print(f"❌ Error in recommend_places: {e}")
        return {"status": "error", "message": str(e)}

# -------------------------------
# 9. Legacy functions (kept for compatibility)
# -------------------------------
def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Legacy function kept for compatibility - not used in new recommendation system"""
    common = set(vec1.keys()) & set(vec2.keys())
    if not common:
        return 0.0
    dot = sum(vec1[k] * vec2[k] for k in common)
    norm1 = sqrt(sum(v**2 for v in vec1.values()))
    norm2 = sqrt(sum(v**2 for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def create_emotion_vector_from_emotions(selected_emotions: List[str]) -> Dict[str, float]:
    """Legacy function kept for compatibility - not used in new recommendation system"""
    emotion_mapping = {
        "Joy/Happy": ["joy", "happy"],
        "Relaxation/Calm": ["calm", "relaxed", "peaceful"],
        "Social Emotion": ["social"],
        "Excitement": ["excitement", "energetic"],
        "Comfort": ["comfort", "cozy"],
        "Adventure": ["adventure", "exploration"],
        "Romance": ["romantic"],
        "Luxury": ["luxury", "premium"],
        "Shopping": ["shopping", "retail"],
        "Nostalgia": ["nostalgia"],
        "Energy": ["energetic", "energy"],
        "Wellness": ["wellness", "health"],
        "Entertainment": ["entertainment"],
        "Exploration": ["exploration", "curious"],
        "Fear": ["fear", "anxiety"],
        "Creativity": ["creative", "artistic"],
        "Spirituality": ["spiritual", "contemplative"],
        "Education": ["educational", "learning"],
        "Retail": ["retail", "shopping"],
        "Outdoors": ["outdoor", "nature"]
    }
    
    vector = {}
    base_weight = 1.0 / len(selected_emotions) if selected_emotions else 0
    
    for emotion in selected_emotions:
        mapped_emotions = emotion_mapping.get(emotion, [emotion.lower()])
        for mapped_emotion in mapped_emotions:
            vector[mapped_emotion] = base_weight
    
    return vector

# -------------------------------
# 10. Additional Context-Aware Testing Endpoints
# -------------------------------
@app.get("/analyze_context/{place_name}")
def analyze_place_context(place_name: str, category: str = "restaurant", 
                          description: str = "", lat: float = None, lon: float = None):
    """
    Analyze the contextual understanding of a place
    """
    try:
        # Create context-aware text
        context_text = emotion_analyzer.create_context_aware_text(place_name, category, description)
        
        # Get emotions
        emotions = emotion_analyzer.predict_emotions_for_place(place_name, category, description)
        
        # If reference data is available, show similarity analysis
        similarity_info = None
        if emotion_analyzer.reference_data is not None:
            place_embedding = emotion_analyzer.model.encode(context_text, convert_to_tensor=True)
            cosine_scores = util.cos_sim(place_embedding, emotion_analyzer.reference_embeddings)[0]
            top_results = torch.topk(cosine_scores, k=5)
            
            similar_places = []
            for score, idx in zip(top_results.values, top_results.indices):
                ref_place = emotion_analyzer.reference_data.iloc[idx.item()]
                similar_places.append({
                    "reference_text": ref_place['text_for_embedding'],
                    "emotion": ref_place['place_emotion'],
                    "similarity_score": round(float(score.item()), 4)
                })
            
            similarity_info = {
                "similar_reference_places": similar_places,
                "avg_similarity": round(float(cosine_scores.mean().item()), 4)
            }
        
        return {
            "place_name": place_name,
            "category": category,
            "original_description": description,
            "context_aware_text": context_text,
            "predicted_emotions": emotions,
            "similarity_analysis": similarity_info,
            "analysis_method": "Context-aware transformer analysis"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/context_stats")
def get_context_analysis_stats():
    """
    Get statistics about the context-aware analysis
    """
    try:
        places = list(places_collection.find({}, {"_id": 0}))
        if not places:
            return {"status": "error", "message": "No places found"}
        
        stats = {
            "total_places": len(places),
            "places_with_descriptions": sum(1 for p in places if p.get("description", "").strip()),
            "places_with_context": sum(1 for p in places if p.get("combined_context", "").strip()),
            "avg_context_length": sum(len(p.get("combined_context", "")) for p in places) / len(places),
            "unique_categories": len(set(p.get("category", "unknown") for p in places)),
            "emotion_distribution": {},
            "context_quality_indicators": {
                "has_detailed_tags": sum(1 for p in places if len(p.get("osm_tags", {})) > 3),
                "has_cuisine_info": sum(1 for p in places if p.get("osm_tags", {}).get("cuisine")),
                "has_opening_hours": sum(1 for p in places if p.get("osm_tags", {}).get("opening_hours")),
                "has_accessibility_info": sum(1 for p in places if p.get("osm_tags", {}).get("wheelchair"))
            }
        }
        
        # Calculate emotion distribution
        all_emotions = {}
        for place in places:
            for emotion_data in place.get("emotions", []):
                emotion = emotion_data.get("emotion")
                if emotion in all_emotions:
                    all_emotions[emotion] += 1
                else:
                    all_emotions[emotion] = 1
        
        stats["emotion_distribution"] = dict(sorted(all_emotions.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "status": "success",
            "analysis_stats": stats,
            "model_info": {
                "model_name": "all-mpnet-base-v2",
                "context_aware": True,
                "reference_data_available": emotion_analyzer.reference_data is not None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# 11. Serve frontend
# -------------------------------
FRONTEND_DIR = r"C:\Users\rhythm\Desktop\recsys\frontend"
if os.path.exists(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

@app.get("/")
def root():
    if os.path.exists(FRONTEND_DIR):
        return RedirectResponse("/frontend/frontpage.html")
    else:
        return {"message": "Enhanced Places Emotion Recommender API with Context-Aware Analysis"}

# -------------------------------
# 12. Auto-open browser
# -------------------------------
def open_frontend():
    time.sleep(3)
    webbrowser.open("http://127.0.0.1:8000/")

# -------------------------------
# 13. Run app
# -------------------------------
if __name__ == "__main__":
    print("🚀 Starting Enhanced Places Emotion Recommender System with Context-Aware Analysis...")
    print("🧠 Features: Name + Description combination, Context-aware transformers, Enhanced emotion mapping")
    threading.Thread(target=open_frontend, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)