package main

import (
	"encoding/json"
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

type UserModel struct {
	UserUUID uuid.UUID `gorm:"primaryKey;type:uuid"`
	Name     string    `gorm:"not null;index"`
}

func (UserModel) TableName() string { return "users" }

type HistoryModel struct {
	HistoryUUID       uuid.UUID `gorm:"primaryKey;type:uuid"`
	UserUUID          uuid.UUID `gorm:"index;not null;type:uuid"` // FK column (we'll add constraint after migrate)
	Age               int
	Height            float64
	Weight            float64
	MedicalConditions string    `gorm:"not null;default:''"`
	CreatedAt         time.Time `gorm:"autoCreateTime"`
}

func (HistoryModel) TableName() string { return "history" }

type CreateUserRequest struct {
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

	// Disable FK creation during AutoMigrate (prevents reverse-FK bug)
	gdb, err := gorm.Open(postgres.Open(dbURL), &gorm.Config{
		Logger:                                   logger.Default.LogMode(logger.Warn),
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

	// Create/align tables
	if err := gdb.AutoMigrate(&UserModel{}, &HistoryModel{}); err != nil {
		log.Fatalf("migrate: %v", err)
	}

	// Add the intended FK (history.user_uuid -> users.user_uuid) only if missing
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

	r := chi.NewRouter()
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins: []string{"http://*", "https://*"},
		AllowedMethods: []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders: []string{"*"},
		MaxAge:         300,
	}))

	r.Get("/health", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("ok")) })
	r.Post("/users", api.handlerCreateUser)
	r.Get("/users/{uuid}/exists", api.handlerCheckExists)
	r.Post("/sync/users", api.handlerSyncUsers)

	log.Printf("[Go API] listening on :%s", port)
	log.Println("GORM/Postgres ready")
	log.Fatal(http.ListenAndServe(":"+port, r))
}

func (a *apiConfig) handlerCreateUser(w http.ResponseWriter, r *http.Request) {
	var req CreateUserRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		httpError(w, http.StatusBadRequest, "invalid json")
		return
	}
	req.Name = strings.TrimSpace(req.Name)
	if req.Name == "" {
		httpError(w, http.StatusBadRequest, "name is required")
		return
	}

	var user UserModel
	err := a.DB.Transaction(func(tx *gorm.DB) error {
		if err := tx.Where("name = ?", req.Name).First(&user).Error; err != nil {
			if err == gorm.ErrRecordNotFound {
				user = UserModel{UserUUID: uuid.New(), Name: req.Name}
				if err := tx.Create(&user).Error; err != nil {
					return err
				}
			} else {
				return err
			}
		} else {
			user.Name = req.Name
			if err := tx.Save(&user).Error; err != nil {
				return err
			}
		}
		h := HistoryModel{
			HistoryUUID: uuid.New(),
			UserUUID:    user.UserUUID,
			Age:         req.Age, Height: req.Height, Weight: req.Weight,
			MedicalConditions: strings.Join(req.MedicalConditions, ", "),
		}
		return tx.Create(&h).Error
	})
	if err != nil {
		httpError(w, http.StatusInternalServerError, "db error")
		return
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
				HistoryUUID: uuid.New(),
				UserUUID:    user.UserUUID,
				Age:         u.Age, Height: u.Height, Weight: u.Weight,
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

func httpError(w http.ResponseWriter, code int, msg string) {
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": msg})
}
