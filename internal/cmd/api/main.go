package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/cors"
	"github.com/google/uuid"
	"github.com/joho/godotenv"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

/* ---------- DB Models ---------- */

type UserModel struct {
	UserUUID uuid.UUID `gorm:"primaryKey;type:uuid"`
	Name     string    `gorm:"not null;index"`
}

func (UserModel) TableName() string { return "users" }

type HistoryModel struct {
	HistoryUUID       uuid.UUID `gorm:"primaryKey;type:uuid"`
	UserUUID          uuid.UUID `gorm:"index;not null;type:uuid"` // FK column
	Age               int
	Height            float64
	Weight            float64
	MedicalConditions string    `gorm:"not null;default:''"`
	CreatedAt         time.Time `gorm:"autoCreateTime"`
}

func (HistoryModel) TableName() string { return "history" }

/* ---------- API Payloads ---------- */

type CreateUserRequest struct {
	UUID              string   `json:"uuid"` // ← client-generated; optional for legacy creates
	Name              string   `json:"name"`
	Age               int      `json:"age"`
	Weight            float64  `json:"weight"`
	Height            float64  `json:"height"`
	MedicalConditions []string `json:"medical_conditions"`
}

type SyncUsersRequest struct {
	Users []struct {
		Name              string   `json:"name"`
		Age               int      `json:"age"`
		Weight            float64  `json:"weight"`
		Height            float64  `json:"height"`
		MedicalConditions []string `json:"medical_conditions"`
	} `json:"users"`
}

// UI queue batch -> /api/sync
type QueueItem struct {
	ID      string          `json:"id"`
	Type    string          `json:"type"` // "createUser" | "updateUser"
	Ts      string          `json:"ts"`
	Payload json.RawMessage `json:"payload"` // shape: CreateUserRequest
}
type SyncBatchRequest struct {
	Items []QueueItem `json:"items"`
}
type SyncBatchResponse struct {
	Synced int `json:"synced"`
	Failed int `json:"failed"`
}

/* ---------- Server ---------- */

type apiConfig struct{ DB *gorm.DB }

func main() {
	_ = godotenv.Load()

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	dbURL := os.Getenv("DB_URL")
	if dbURL == "" {
		log.Fatal("DB_URL is not set")
	}

	// Connect DB
	gdb, err := gorm.Open(postgres.Open(dbURL), &gorm.Config{
		Logger: logger.New(
			log.New(os.Stdout, "\r\n", log.LstdFlags),
			logger.Config{
				LogLevel:                  logger.Error, // only errors
				IgnoreRecordNotFoundError: true,         // don't log ErrRecordNotFound
			},
		),
		DisableForeignKeyConstraintWhenMigrating: true,
	})

	if err != nil {
		log.Fatalf("connect db: %v", err)
	}
	sqlDB, err := gdb.DB()
	if err != nil {
		log.Fatalf("db(): %v", err)
	}
	if err := sqlDB.Ping(); err != nil {
		log.Fatalf("ping: %v", err)
	}

	// Migrate
	if err := gdb.AutoMigrate(&UserModel{}, &HistoryModel{}); err != nil {
		log.Fatalf("migrate: %v", err)
	}
	// Add FK if missing
	if err := gdb.Exec(`
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_history_useruuid'
    ) THEN
        ALTER TABLE history
            ADD CONSTRAINT fk_history_useruuid
            FOREIGN KEY (user_uuid)
            REFERENCES users(user_uuid)
            ON UPDATE CASCADE
            ON DELETE CASCADE;
    END IF;
END$$;
`).Error; err != nil {
		log.Printf("[WARN] adding FK failed: %v", err)
	}

	api := &apiConfig{DB: gdb}

	// Router + CORS
	r := chi.NewRouter()
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins: []string{"http://*", "https://*"},
		AllowedMethods: []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders: []string{"*"},
		MaxAge:         300,
	}))

	// Routes
	r.Get("/health", func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte("ok")) })
	r.Post("/users", api.handlerCreateUser)
	r.Get("/users/{uuid}/exists", api.handlerCheckExists)
	r.Post("/sync/users", api.handlerSyncUsers) // legacy/alternate
	r.Post("/api/sync", api.handlerSync)        // UI queue posts here

	log.Printf("[Go API] listening on :%s", port)
	log.Println("GORM/Postgres ready")
	log.Fatal(http.ListenAndServe(":"+port, r))
}

/* ---------- Helpers ---------- */

// upsertUserHistory prefers client UUID (if provided). If UUID is present:
//   - find by UUID; update name (if provided) and append history
//   - if not found, create user with that UUID
//
// If UUID is empty:
//   - upsert by name (legacy behavior)
func (a *apiConfig) upsertUserHistory(req CreateUserRequest) error {
	cleanName := strings.TrimSpace(req.Name)
	hasUUID := strings.TrimSpace(req.UUID) != ""

	return a.DB.Transaction(func(tx *gorm.DB) error {
		var user UserModel
		var err error

		if hasUUID {
			uid, parseErr := uuid.Parse(req.UUID)
			if parseErr != nil {
				return fmt.Errorf("invalid uuid: %w", parseErr)
			}
			err = tx.First(&user, "user_uuid = ?", uid).Error
			if err != nil {
				if err == gorm.ErrRecordNotFound {
					// create with client-provided UUID
					if cleanName == "" {
						return fmt.Errorf("name is required when creating a user")
					}
					user = UserModel{UserUUID: uid, Name: cleanName}
					if err := tx.Create(&user).Error; err != nil {
						return err
					}
				} else {
					return err
				}
			} else {
				// update name if provided
				if cleanName != "" && user.Name != cleanName {
					user.Name = cleanName
					if err := tx.Save(&user).Error; err != nil {
						return err
					}
				}
			}
		} else {
			// legacy path: upsert by name
			if cleanName == "" {
				return fmt.Errorf("name is required")
			}
			err = tx.Where("name = ?", cleanName).First(&user).Error
			if err != nil {
				if err == gorm.ErrRecordNotFound {
					user = UserModel{UserUUID: uuid.New(), Name: cleanName}
					if err := tx.Create(&user).Error; err != nil {
						return err
					}
				} else {
					return err
				}
			} else {
				if user.Name != cleanName {
					user.Name = cleanName
					if err := tx.Save(&user).Error; err != nil {
						return err
					}
				}
			}
		}

		// Append history snapshot
		h := HistoryModel{
			HistoryUUID:       uuid.New(),
			UserUUID:          user.UserUUID,
			Age:               req.Age,
			Height:            req.Height,
			Weight:            req.Weight,
			MedicalConditions: strings.Join(req.MedicalConditions, ", "),
		}
		return tx.Create(&h).Error
	})
}

/* ---------- Handlers ---------- */

func (a *apiConfig) handlerCreateUser(w http.ResponseWriter, r *http.Request) {
	var req CreateUserRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpError(w, http.StatusBadRequest, "invalid json")
		return
	}
	if err := a.upsertUserHistory(req); err != nil {
		httpError(w, http.StatusBadRequest, err.Error())
		return
	}

	// Return the user's UUID. We must read it by UUID (if provided) or by name.
	var user UserModel
	if strings.TrimSpace(req.UUID) != "" {
		uid, _ := uuid.Parse(req.UUID)
		if err := a.DB.First(&user, "user_uuid = ?", uid).Error; err != nil {
			httpError(w, http.StatusInternalServerError, "db error")
			return
		}
	} else {
		if err := a.DB.Where("name = ?", strings.TrimSpace(req.Name)).First(&user).Error; err != nil {
			httpError(w, http.StatusInternalServerError, "db error")
			return
		}
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"uuid": user.UserUUID})
}

func (a *apiConfig) handlerCheckExists(w http.ResponseWriter, r *http.Request) {
	param := chi.URLParam(r, "uuid")
	if param == "" {
		httpError(w, http.StatusBadRequest, "uuid required")
		return
	}
	uid, err := uuid.Parse(param)
	if err != nil {
		httpError(w, http.StatusBadRequest, "invalid uuid")
		return
	}
	var count int64
	if err := a.DB.Model(&UserModel{}).Where("user_uuid = ?", uid).Count(&count).Error; err != nil {
		httpError(w, http.StatusInternalServerError, "db error")
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]bool{"exists": count > 0})
}

func (a *apiConfig) handlerSyncUsers(w http.ResponseWriter, r *http.Request) {
	var req SyncUsersRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpError(w, http.StatusBadRequest, "invalid json")
		return
	}
	if len(req.Users) == 0 {
		_ = json.NewEncoder(w).Encode(map[string]any{"upserted": 0})
		return
	}

	var total int64
	err := a.DB.Transaction(func(tx *gorm.DB) error {
		for _, u := range req.Users {
			name := strings.TrimSpace(u.Name)
			if name == "" {
				continue
			}

			var user UserModel
			err := tx.Where("name = ?", name).First(&user).Error
			if err != nil && err != gorm.ErrRecordNotFound {
				return err
			}
			if err == gorm.ErrRecordNotFound {
				user = UserModel{UserUUID: uuid.New(), Name: name}
				if err := tx.Create(&user).Error; err != nil {
					return err
				}
			} else {
				user.Name = name
				if err := tx.Save(&user).Error; err != nil {
					return err
				}
			}

			h := HistoryModel{
				HistoryUUID:       uuid.New(),
				UserUUID:          user.UserUUID,
				Age:               u.Age,
				Height:            u.Height,
				Weight:            u.Weight,
				MedicalConditions: strings.Join(u.MedicalConditions, ", "),
			}
			if err := tx.Create(&h).Error; err != nil {
				return err
			}
			total++
		}
		return nil
	})
	if err != nil {
		httpError(w, http.StatusInternalServerError, "db error")
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"upserted": total})
}

func (a *apiConfig) handlerSync(w http.ResponseWriter, r *http.Request) {
	var batch SyncBatchRequest
	if err := json.NewDecoder(r.Body).Decode(&batch); err != nil {
		httpError(w, http.StatusBadRequest, "invalid json")
		return
	}
	if len(batch.Items) == 0 {
		_ = json.NewEncoder(w).Encode(SyncBatchResponse{Synced: 0, Failed: 0})
		return
	}

	synced, failed := 0, 0
	for _, it := range batch.Items {
		if it.Type != "createUser" && it.Type != "updateUser" {
			failed++
			continue
		}
		var u CreateUserRequest
		if err := json.Unmarshal(it.Payload, &u); err != nil {
			failed++
			continue
		}
		if err := a.upsertUserHistory(u); err != nil {
			failed++
			continue
		}
		synced++
	}

	_ = json.NewEncoder(w).Encode(SyncBatchResponse{Synced: synced, Failed: failed})
}

/* ---------- Util ---------- */

func httpError(w http.ResponseWriter, code int, msg string) {
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": msg})
}
