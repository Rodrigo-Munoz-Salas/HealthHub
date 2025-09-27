package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/google/uuid"
)

type User struct {
	Name              string   `json:"name"`
	Age               int      `json:"age"`
	Weight            float64  `json:"weight"`
	Height            float64  `json:"height"`
	MedicalConditions []string `json:"medical_conditions"`
}

var (
	users   = make(map[string]User)
	usersMu sync.RWMutex
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	})

	// Simple CORS
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if origin := r.Header.Get("Origin"); origin != "" {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Vary", "Origin")
		}
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		path := r.URL.Path
		if path == "/users" && r.Method == http.MethodPost {
			createUser(w, r)
			return
		}
		if strings.HasPrefix(path, "/users/") && strings.HasSuffix(path, "/exists") && r.Method == http.MethodGet {
			checkExists(w, r)
			return
		}

		http.NotFound(w, r)
	})

	log.Printf("[Go API] listening on :%s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func createUser(w http.ResponseWriter, r *http.Request) {
	var u User
	if err := json.NewDecoder(r.Body).Decode(&u); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "invalid json"})
		return
	}
	log.Printf("[Go API] User information received! Data: %+v", u)

	id := uuid.New().String()
	usersMu.Lock()
	users[id] = u
	usersMu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"uuid": id})
}

func checkExists(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
	if len(parts) < 3 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "invalid path"})
		return
	}
	id := parts[1]

	usersMu.RLock()
	_, ok := users[id]
	usersMu.RUnlock()

	log.Printf("[Go API] Verify called for uuid=%s => exists=%v", id, ok)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"exists": ok})
}
