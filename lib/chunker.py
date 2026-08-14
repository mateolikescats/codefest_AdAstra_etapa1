import re
from typing import List, Dict, Any

class SentenceChunker:
    def __init__(self, max_words: int = 240, target_words: int = 200, overlap_words: int = 40):
        self.max_words = max_words
        self.target_words = target_words
        self.overlap_words = overlap_words

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Splits raw text into linguistically complete sentences.
        Preserves sentence terminators (. ! ? \n).
        """
        if not text or not text.strip():
            return []
        
        # Split on sentence boundaries (. ! ? or double newlines)
        sentence_end = re.compile(r'(?<=[.!?])\s+|\n\n+')
        raw_sentences = sentence_end.split(text)
        
        sentences = []
        for s in raw_sentences:
            s_clean = s.strip()
            if s_clean:
                sentences.append(s_clean)
                
        return sentences

    def chunk_document(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunks a document into sentence-boundary aligned chunks of <= 250 words.
        Attaches all mandatory metadata fields (doc_id, chunk_id, fuente, formato, fenomeno, posicion, num_tokens, texto).
        """
        doc_id = doc_data["doc_id"]
        text = doc_data["text"]
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_word_count = 0
        chunk_pos = 0

        for sentence in sentences:
            words_in_sent = len(sentence.split())
            
            # If a single sentence exceeds max_words, split it by clause boundaries
            if words_in_sent > self.max_words:
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(self._build_chunk_dict(doc_data, chunk_text, chunk_pos))
                    chunk_pos += 1
                    current_sentences = []
                    current_word_count = 0
                
                # Split long sentence by clauses (commas, semicolons, colons)
                clause_chunks = self._split_long_sentence(sentence, self.max_words)
                for c_text in clause_chunks:
                    chunks.append(self._build_chunk_dict(doc_data, c_text, chunk_pos))
                    chunk_pos += 1
                continue

            if current_word_count + words_in_sent > self.target_words and current_word_count > 0:
                # Finish current chunk
                chunk_text = " ".join(current_sentences)
                chunks.append(self._build_chunk_dict(doc_data, chunk_text, chunk_pos))
                chunk_pos += 1
                
                # Apply sliding window overlap by keeping last sentences up to overlap_words
                overlap_sentences = []
                overlap_count = 0
                for s in reversed(current_sentences):
                    w_cnt = len(s.split())
                    if overlap_count + w_cnt <= self.overlap_words:
                        overlap_sentences.insert(0, s)
                        overlap_count += w_cnt
                    else:
                        break
                        
                current_sentences = overlap_sentences
                current_word_count = overlap_count

            current_sentences.append(sentence)
            current_word_count += words_in_sent

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            # Guarantee chunk is strictly <= 250 words
            if len(chunk_text.split()) > self.max_words:
                chunk_text = " ".join(chunk_text.split()[:self.max_words])
            chunks.append(self._build_chunk_dict(doc_data, chunk_text, chunk_pos))

        return chunks

    def _split_long_sentence(self, sentence: str, max_words: int) -> List[str]:
        """
        Splits a long sentence by clause boundaries (;, :, ,) while staying under max_words.
        """
        # First try splitting by clause punctuation
        clauses = re.split(r'(?<=[;,:])\s+', sentence)
        sub_chunks = []
        curr_words = []
        curr_cnt = 0

        for clause in clauses:
            c_words = clause.split()
            if curr_cnt + len(c_words) <= max_words:
                curr_words.extend(c_words)
                curr_cnt += len(c_words)
            else:
                if curr_words:
                    sub_chunks.append(" ".join(curr_words))
                if len(c_words) > max_words:
                    # Fallback to word slicing if a single clause is huge
                    for i in range(0, len(c_words), max_words):
                        sub_chunks.append(" ".join(c_words[i:i + max_words]))
                    curr_words = []
                    curr_cnt = 0
                else:
                    curr_words = list(c_words)
                    curr_cnt = len(c_words)
        if curr_words:
            sub_chunks.append(" ".join(curr_words))

        return sub_chunks if sub_chunks else [sentence]

    def _build_chunk_dict(self, doc_data: Dict[str, Any], text: str, pos: int) -> Dict[str, Any]:
        words = text.split()
        num_words = len(words)
        doc_id = doc_data["doc_id"]
        chunk_id = f"{doc_id}-chunk-{pos:04d}"
        
        return {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "fuente": doc_data["fuente"],
            "formato": doc_data["formato"],
            "fenomeno": doc_data["fenomeno"],
            "posicion": pos,
            "num_tokens": num_words,
            "texto": text
        }

