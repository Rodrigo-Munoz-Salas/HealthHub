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

type SyncUsersRequest struct {
	Users []struct {
		UUID              string   `json:"uuid"`
		Name              string   `json:"name"`
		Age               int      `json:"age"`
		Weight            float64  `json:"weight"`
		Height            float64  `json:"height"`
		MedicalConditions []string `json:"medical_conditions"`
	} `json:"users"`
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	// Single handler with permissive CORS (dev)
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

		switch {
		case r.URL.Path == "/users" && r.Method == http.MethodPost:
			createUser(w, r)
			return
		case strings.HasPrefix(r.URL.Path, "/users/") && strings.HasSuffix(r.URL.Path, "/exists") && r.Method == http.MethodGet:
			checkExists(w, r)
			return
		case r.URL.Path == "/sync/users" && r.Method == http.MethodPost:
			syncUsers(w, r)
			return
		default:
			http.NotFound(w, r)
		}
	})

	log.Printf("[Go API] listening on :%s", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func createUser(w http.ResponseWriter, r *http.Request) {
	var u User
	if err := json.NewDecoder(r.Body).Decode(&u); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "invalid json"})
		return
	}
	log.Printf("[Go API] User information received! Data: %+v", u)

	id := uuid.New().String()
	usersMu.Lock()
	users[id] = u
	usersMu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"uuid": id})
}

func checkExists(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
	if len(parts) < 3 {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "invalid path"})
		return
	}
	id := parts[1]

	usersMu.RLock()
	_, ok := users[id]
	usersMu.RUnlock()

	log.Printf("[Go API] Verify called for uuid=%s => exists=%v", id, ok)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]bool{"exists": ok})
}

func syncUsers(w http.ResponseWriter, r *http.Request) {
	var req SyncUsersRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": "invalid json"})
		return
	}

	added := 0
	for _, u := range req.Users {
		if u.UUID == "" {
			continue
		}
		usersMu.Lock()
		users[u.UUID] = User{
			Name: u.Name, Age: u.Age,
			Weight: u.Weight, Height: u.Height,
			MedicalConditions: u.MedicalConditions,
		}
		usersMu.Unlock()
		added++
	}

	log.Printf("[Go API] /sync/users upserted=%d", added)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"upserted": added})
}
