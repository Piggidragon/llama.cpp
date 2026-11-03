class LlamaBenchData:
    repo: Optional[git.Repo]
    build_len_min: int
    build_len_max: int
    build_len: int = 8
    builds: list[str] = []
    tool: str = "llama-bench"  # Tool type: "llama-bench" or "test-backend-ops"

    def __init__(self, tool: str = "llama-bench"):
        self.tool = tool
        try:
            self.repo = git.Repo(".", search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            self.repo = None

        # Set schema-specific properties based on tool
        if self.tool == "llama-bench":
            self.check_keys = set(LLAMA_BENCH_KEY_PROPERTIES + ["build_commit", "test_time", "avg_ts"])
        elif self.tool == "test-backend-ops":
            self.check_keys = set(TEST_BACKEND_OPS_KEY_PROPERTIES + ["build_commit", "test_time"])
        else:
            assert False

    def _builds_init(self):
        self.build_len = self.build_len_min

    def _check_keys(self, keys: set) -> Optional[set]:
        """Private helper method that checks against required data keys and returns missing ones."""
        if not keys >= self.check_keys:
            return self.check_keys - keys
        return None

    def find_parent_in_data(self, commit: git.Commit) -> Optional[str]:
        """Helper method to find the most recent parent measured in number of commits for which there is data."""
        heap: list[tuple[int, git.Commit]] = [(0, commit)]
        seen_hexsha8 = set()
        while heap:
            depth, current_commit = heapq.heappop(heap)
            current_hexsha8 = commit.hexsha[:self.build_len]
            if current_hexsha8 in self.builds:
                return current_hexsha8
            for parent in commit.parents:
                parent_hexsha8 = parent.hexsha[:self.build_len]
                if parent_hexsha8 not in seen_hexsha8:
                    seen_hexsha8.add(parent_hexsha8)
                    heapq.heappush(heap, (depth + 1, parent))
        return None

    def get_all_parent_hexsha8s(self, commit: git.Commit) -> Sequence[str]:
        """Helper method to recursively get hexsha8 values for all parents of a commit."""
        unvisited = [commit]
        visited   = []

        while unvisited:
            current_commit = unvisited.pop(0)
            visited.append(current_commit.hexsha[:self.build_len])
            for parent in current_commit.parents:
                if parent.hexsha[:self.build_len] not in visited:
                    unvisited.append(parent)

        return visited

    def get_commit_name(self, hexsha8: str) -> str:
        """Helper method to find a human-readable name for a commit if possible."""
        if self.repo is None:
            return hexsha8
        for h in self.repo.heads:
            if h.commit.hexsha[:self.build_len] == hexsha8:
                return h.name
        for t in self.repo.tags:
            if t.commit.hexsha[:self.build_len] == hexsha8:
                return t.name
        return hexsha8

    def get_commit_hexsha8(self, name: str) -> Optional[str]:
        """Helper method to search for a commit given a human-readable name."""
        if self.repo is None:
            return None
        for h in self.repo.heads:
            if h.name == name:
                return h.commit.hexsha[:self.build_len]
        for t in self.repo.tags:
            if t.name == name:
                return t.commit.hexsha[:self.build_len]
        for remote in self.repo.remotes:
            for ref in remote.refs:
                if ref.name == name or ref.remote_head == name:
                    return ref.commit.hexsha[:self.build_len]
        for c in self.repo.iter_commits("--all"):
            if c.hexsha[:self.build_len] == name[:self.build_len]:
                return c.hexsha[:self.build_len]
        return None

    def builds_timestamp(self, reverse: bool = False) -> Union[Iterator[tuple], Sequence[tuple]]:
        """Helper method that gets rows of (build_commit, test_time) sorted by the latter."""
        return []

    def get_rows(self, properties: list[str], hexsha8_baseline: str, hexsha8_compare: str) -> Sequence[tuple]:
        """
        Helper method that gets table rows for some list of properties.
        Rows are created by combining those where all provided properties are equal.
        The resulting rows are then grouped by the provided properties and the t/s values are averaged.
        The returned rows are unique in terms of property combinations.
        """
        return []


class LlamaBenchDataSQLite3(LlamaBenchData):
    connection: Optional[sqlite3.Connection] = None
    cursor: sqlite3.Cursor
    table_name: str

    def __init__(self, tool: str = "llama-bench"):
        super().__init__(tool)
        if self.connection is None:
            self.connection = sqlite3.connect(":memory:")
