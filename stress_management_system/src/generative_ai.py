"""
Generative AI Module with RAG (Retrieval-Augmented Generation)
Provides personalized stress management recommendations
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import warnings
warnings.filterwarnings('ignore')


class StressManagementRAG:
    """
    RAG-based recommendation system for stress management
    """
    
    def __init__(self, knowledge_base_path, embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize RAG system
        
        Args:
            knowledge_base_path: Path to knowledge base directory
            embedding_model: Sentence transformer model for embeddings
        """
        self.knowledge_base_path = knowledge_base_path
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.collection = None
        self.client = None
        
    def initialize(self):
        """Initialize embedding model and vector database"""
        print("Initializing RAG system...")
        
        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize ChromaDB (updated for newer versions)
        print("Initializing vector database...")
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection("stress_management")
            print("Loaded existing collection")
        except:
            self.collection = self.client.create_collection("stress_management")
            print("Created new collection")
            self._load_knowledge_base()
        
        print("RAG system initialized")
    
    def _load_knowledge_base(self):
        """Load knowledge base documents into vector database"""
        print("Loading knowledge base documents...")
        
        # Read markdown files
        documents = []
        metadatas = []
        ids = []
        
        for filename in os.listdir(self.knowledge_base_path):
            if filename.endswith('.md'):
                filepath = os.path.join(self.knowledge_base_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split into sections
                sections = self._split_into_sections(content)
                
                for i, section in enumerate(sections):
                    documents.append(section['content'])
                    metadatas.append({
                        'source': filename,
                        'title': section['title'],
                        'category': section['category']
                    })
                    ids.append(f"{filename}_{i}")
        
        if len(documents) == 0:
            print("Warning: No documents found in knowledge base")
            return
        
        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} sections...")
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Loaded {len(documents)} sections into vector database")
    
    def _split_into_sections(self, content):
        """
        Split markdown content into sections
        
        Args:
            content: Markdown content
            
        Returns:
            List of section dictionaries
        """
        sections = []
        current_section = {'title': '', 'content': '', 'category': ''}
        current_category = ''
        
        lines = content.split('\n')
        
        for line in lines:
            if line.startswith('## '):
                # Save previous section
                if current_section['content']:
                    sections.append(current_section.copy())
                
                # Start new section
                current_category = line.replace('## ', '').strip()
                current_section = {
                    'title': current_category,
                    'content': line + '\n',
                    'category': current_category
                }
            elif line.startswith('### '):
                # Subsection
                if current_section['content'] and not current_section['content'].startswith('## '):
                    sections.append(current_section.copy())
                
                title = line.replace('### ', '').strip()
                current_section = {
                    'title': title,
                    'content': line + '\n',
                    'category': current_category
                }
            else:
                current_section['content'] += line + '\n'
        
        # Add last section
        if current_section['content']:
            sections.append(current_section)
        
        return sections
    
    def retrieve_recommendations(self, query, top_k=3):
        """
        Retrieve relevant recommendations based on query
        
        Args:
            query: Query string
            top_k: Number of top results to retrieve
            
        Returns:
            List of relevant document sections
        """
        if self.collection is None:
            raise ValueError("RAG system not initialized. Call initialize() first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Query collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        recommendations = []
        for i in range(len(results['documents'][0])):
            recommendations.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return recommendations
    
    def generate_personalized_advice(self, stress_level, contributing_factors, context=None):
        """
        Generate personalized stress management advice
        
        Args:
            stress_level: Stress level (0-1 or 'LOW', 'MODERATE', 'HIGH')
            contributing_factors: List of contributing factors from XAI
            context: Additional context (time of day, work environment, etc.)
            
        Returns:
            Personalized advice string
        """
        # Convert stress level to text
        if isinstance(stress_level, (int, float)):
            if stress_level > 0.7:
                stress_text = "HIGH"
            elif stress_level > 0.4:
                stress_text = "MODERATE"
            else:
                stress_text = "LOW"
        else:
            stress_text = stress_level
        
        # Build query based on contributing factors
        queries = []
        
        # Analyze contributing factors
        has_high_hrv = any('hrv' in str(f).lower() for f in contributing_factors)
        has_high_eda = any('eda' in str(f).lower() for f in contributing_factors)
        has_high_hr = any('hr' in str(f).lower() or 'ecg' in str(f).lower() for f in contributing_factors)
        has_resp_issues = any('resp' in str(f).lower() for f in contributing_factors)
        
        # Create targeted queries
        if has_high_eda or has_high_hr:
            queries.append("immediate stress relief breathing techniques")
        
        if has_high_hrv:
            queries.append("heart rate variability stress management meditation")
        
        if has_resp_issues:
            queries.append("breathing exercises diaphragmatic breathing")
        
        if stress_text == "HIGH":
            queries.append("emergency stress relief quick techniques")
            queries.append("when to seek professional help")
        elif stress_text == "MODERATE":
            queries.append("work stress management time management")
            queries.append("mindfulness meditation")
        else:
            queries.append("stress prevention work-life balance")
        
        # Retrieve recommendations
        all_recommendations = []
        for query in queries[:3]:  # Limit to top 3 queries
            recs = self.retrieve_recommendations(query, top_k=2)
            all_recommendations.extend(recs)
        
        # Remove duplicates
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            content_hash = hash(rec['content'])
            if content_hash not in seen:
                seen.add(content_hash)
                unique_recommendations.append(rec)
        
        # Generate advice
        advice = self._format_advice(stress_text, unique_recommendations[:5], contributing_factors)
        
        return advice
    
    def _format_advice(self, stress_level, recommendations, contributing_factors):
        """
        Format recommendations into readable advice
        
        Args:
            stress_level: Stress level text
            recommendations: Retrieved recommendations
            contributing_factors: Contributing factors
            
        Returns:
            Formatted advice string
        """
        advice = f"🧘 Personalized Stress Management Recommendations\n"
        advice += f"{'='*60}\n\n"
        advice += f"Detected Stress Level: {stress_level}\n\n"
        
        # Immediate actions for high stress
        if stress_level == "HIGH":
            advice += "⚠️ IMMEDIATE ACTIONS RECOMMENDED:\n\n"
            advice += "1. **Stop current activity** - Take a break immediately\n"
            advice += "2. **Practice Box Breathing** - 4 counts in, hold 4, out 4, hold 4\n"
            advice += "3. **5-4-3-2-1 Grounding** - Notice 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste\n"
            advice += "4. **Move your body** - Take a quick 5-minute walk\n\n"
        
        # Main recommendations
        advice += "📋 RECOMMENDED TECHNIQUES:\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            content = rec['content'].strip()
            category = rec['metadata'].get('category', 'General')
            
            # Extract actionable parts
            lines = content.split('\n')
            actionable = []
            for line in lines:
                if line.startswith('**') or line.startswith('-') or line.startswith('1.'):
                    actionable.append(line)
            
            if actionable:
                advice += f"\n{i}. **{category}**\n"
                for action in actionable[:5]:  # Limit to 5 actions per recommendation
                    advice += f"   {action}\n"
        
        # Physiological-specific advice
        advice += "\n\n💡 BASED ON YOUR PHYSIOLOGICAL SIGNALS:\n\n"
        
        has_eda = any('eda' in str(f).lower() for f in contributing_factors)
        has_hrv = any('hrv' in str(f).lower() for f in contributing_factors)
        has_resp = any('resp' in str(f).lower() for f in contributing_factors)
        
        if has_eda:
            advice += "- Your skin conductance is elevated, indicating sympathetic activation\n"
            advice += "  → Focus on **cooling techniques**: cold water on wrists, cool environment\n\n"
        
        if has_hrv:
            advice += "- Your heart rate variability shows stress response\n"
            advice += "  → Practice **HRV-focused breathing**: 5-6 breaths per minute\n\n"
        
        if has_resp:
            advice += "- Your respiration pattern indicates tension\n"
            advice += "  → Try **diaphragmatic breathing**: breathe into belly, not chest\n\n"
        
        # Long-term recommendations
        advice += "\n📅 LONG-TERM STRATEGIES:\n\n"
        advice += "- Maintain regular sleep schedule (7-9 hours)\n"
        advice += "- Practice daily mindfulness or meditation (10-15 minutes)\n"
        advice += "- Exercise regularly (30 minutes, 5 days/week)\n"
        advice += "- Limit caffeine and alcohol\n"
        advice += "- Set boundaries for work hours\n"
        advice += "- Stay connected with friends and family\n"
        
        # Professional help recommendation
        if stress_level == "HIGH":
            advice += "\n\n⚕️ CONSIDER PROFESSIONAL SUPPORT:\n"
            advice += "If stress persists or worsens, consult:\n"
            advice += "- Mental health professional\n"
            advice += "- Employee Assistance Program (EAP)\n"
            advice += "- Primary care physician\n"
        
        advice += f"\n{'='*60}\n"
        
        return advice
    
    def get_quick_tip(self, stress_level):
        """
        Get a quick stress management tip
        
        Args:
            stress_level: Stress level
            
        Returns:
            Quick tip string
        """
        tips = {
            "HIGH": [
                "🌬️ Try box breathing RIGHT NOW: Breathe in for 4, hold for 4, out for 4, hold for 4. Repeat 5 times.",
                "❄️ Splash cold water on your face or hold an ice cube. This activates the dive reflex and calms your nervous system.",
                "🚶 Take an immediate 5-minute walk. Movement helps process stress hormones.",
                "📱 Step away from screens. Look at something 20 feet away for 20 seconds."
            ],
            "MODERATE": [
                "🧘 Take a 5-minute mindfulness break. Close your eyes and focus on your breath.",
                "💧 Drink a glass of water. Dehydration can amplify stress responses.",
                "🎵 Listen to calming music for 10 minutes.",
                "✍️ Write down 3 things you're grateful for right now."
            ],
            "LOW": [
                "🌟 Great job managing stress! Maintain this with daily 10-minute meditation.",
                "💪 Keep up good habits: regular exercise, good sleep, healthy eating.",
                "🤝 Stay connected with friends and colleagues.",
                "📚 Consider learning a new stress management technique to add to your toolkit."
            ]
        }
        
        import random
        stress_key = stress_level if stress_level in tips else "MODERATE"
        return random.choice(tips[stress_key])


if __name__ == "__main__":
    # Example usage
    print("Setting up RAG system...")
    
    rag = StressManagementRAG(knowledge_base_path="../knowledge_base")
    rag.initialize()
    
    # Test retrieval
    print("\nTesting recommendation retrieval...")
    query = "breathing exercises for immediate stress relief"
    recommendations = rag.retrieve_recommendations(query, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(recommendations)} recommendations\n")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. Category: {rec['metadata']['category']}")
        print(f"   Title: {rec['metadata']['title']}")
        print(f"   Distance: {rec['distance']:.4f}")
        print()
    
    # Test personalized advice
    print("\nGenerating personalized advice...")
    contributing_factors = ['eda_mean', 'hrv_rmssd', 'ecg_mean']
    advice = rag.generate_personalized_advice(
        stress_level=0.85,
        contributing_factors=contributing_factors
    )
    
    print(advice)
    
    # Quick tip
    print("\nQuick tip:")
    print(rag.get_quick_tip("HIGH"))
