package plugin

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
)

const Name = "CongestionAware"

type NodeScore struct {
	NodeName    string    `json:"node_name"`
	Score       float64   `json:"score"`
	LastUpdated time.Time `json:"last_updated"`
}

type Args struct {
	metav1.TypeMeta `json:",inline"`

	CacheURL           string        `json:"cacheURL"`
	SyncInterval       time.Duration `json:"syncInterval"`
	StalenessThreshold time.Duration `json:"stalenessThreshold"`
}

func (a *Args) DeepCopyObject() runtime.Object {
	if a == nil {
		return nil
	}
	out := new(Args)
	*out = *a
	out.TypeMeta = a.TypeMeta
	return out
}

type CongestionAwarePlugin struct {
	handle fwk.Handle

	mu         sync.RWMutex
	nodeScores map[string]NodeScore

	client             *http.Client
	cacheURL           string
	stalenessThreshold time.Duration
}

var _ fwk.ScorePlugin = &CongestionAwarePlugin{}

func New(_ context.Context, obj runtime.Object, h fwk.Handle) (fwk.Plugin, error) {
	args, ok := obj.(*Args)
	if !ok {
		return nil, fmt.Errorf("want *Args, got %T", obj)
	}
	if args.SyncInterval == 0 {
		args.SyncInterval = 2 * time.Second
	}
	if args.StalenessThreshold == 0 {
		args.StalenessThreshold = 2 * 60 * time.Second
	}

	p := &CongestionAwarePlugin{
		handle:             h,
		nodeScores:         make(map[string]NodeScore),
		client:             &http.Client{Timeout: 1500 * time.Millisecond},
		cacheURL:           args.CacheURL,
		stalenessThreshold: args.StalenessThreshold,
	}

	ctx := context.Background()

	p.pullFromCacheAPI(ctx)
	go p.syncLoop(ctx, args.SyncInterval)

	return p, nil
}

func (p *CongestionAwarePlugin) Name() string { return Name }

func (p *CongestionAwarePlugin) syncLoop(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.pullFromCacheAPI(ctx)
		}
	}
}

func (p *CongestionAwarePlugin) pullFromCacheAPI(ctx context.Context) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.cacheURL+"/scores", nil)
	if err != nil {
		klog.ErrorS(err, "failed to build cache API request")
		return
	}

	resp, err := p.client.Do(req)
	if err != nil {
		klog.Warningf("cache API unreachable, serving stale local scores: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		klog.Warningf("cache API returned status %d", resp.StatusCode)
		return
	}

	var fresh map[string]NodeScore
	if err := json.NewDecoder(resp.Body).Decode(&fresh); err != nil {
		klog.ErrorS(err, "failed to decode cache API response")
		return
	}

	p.mu.Lock()
	p.nodeScores = fresh
	p.mu.Unlock()

	klog.V(4).Infof("synced %d node scores from cache API", len(fresh))
}

func (p *CongestionAwarePlugin) Score(ctx context.Context, state fwk.CycleState, pod *v1.Pod, node fwk.NodeInfo) (int64, *fwk.Status) {
	nodeName := node.Node().Name
	p.mu.RLock()
	ns, ok := p.nodeScores[nodeName]
	p.mu.RUnlock()

	if !ok {
		klog.V(4).Infof("no score for node %s, using neutral score", nodeName)
		return 50, fwk.NewStatus(fwk.Success)
	}

	if age := time.Since(ns.LastUpdated); age > p.stalenessThreshold {
		klog.Warningf("stale score for node %s, age=%v, falling back to neutral", nodeName, age)
		return 50, fwk.NewStatus(fwk.Success)
	}

	return int64(ns.Score * 100), fwk.NewStatus(fwk.Success)
}

func (p *CongestionAwarePlugin) ScoreExtensions() fwk.ScoreExtensions {
	return nil
}
